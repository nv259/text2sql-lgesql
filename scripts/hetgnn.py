#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:], task='hetgnn')
exp_path = hyperparam_path(args, task='hetgnn')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
logger = set_logger(exp_path, args.testing)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
if args.read_model_path:
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    args.ptm, args.model, args.add_cls, args.lazy_load = params.ptm, params.model, params.add_cls, True
# set up the grammar, transition system, evaluator, etc.
Example.configuration(ptm=args.ptm, method=args.model, add_cls=args.add_cls)
train_dataset, dev_dataset = Example.load_dataset('train'), Example.load_dataset('dev')
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init, set optimizer
if args.read_model_path:
    model = Registrable.by_name('hetgnn-sql')(params, sql_trans).to(device)
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'))
    model.load_state_dict(check_point['model'])
    logger.info("Load saved model from path: %s" % (args.read_model_path))
else:
    json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
    model = Registrable.by_name('hetgnn-sql')(args, sql_trans).to(device)
    if args.ptm is None:
        ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
        logger.info("Init model and word embedding layer with a coverage %.2f" % (ratio))
# logger.info(str(model))
torch.save(model.state_dict(), open(os.path.join(exp_path, 'model.bin'), 'wb'))

def decode(choice, output_path, acc_type='sql'):
    assert acc_type in ['beam', 'ast', 'sql'] and choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    all_hyps = []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False, method='hetgnn')
            hyps = model.parse(current_batch, args.beam_size)
            all_hyps.extend(hyps)
        acc = evaluator.acc(all_hyps, dataset, output_path, acc_type=acc_type, etype='match')
    torch.cuda.empty_cache()
    gc.collect()
    return acc

if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
    nsamples, best_result = len(train_dataset), {'dev_acc': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
    logger.info('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss, count = 0, 0
        np.random.shuffle(train_index)
        model.train()
        for j in range(0, nsamples, step_size):
            count += 1
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, method='hetgnn')
            loss = model(current_batch) # see utils/batch.py for batch elements
            epoch_loss += loss.item()
            # print("Minibatch loss: %.4f" % (loss.item()))
            loss.backward()
            if count == args.grad_accumulate or j + step_size >= nsamples:
                count = 0
                model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f' % (i, time.time() - start_time, epoch_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if i <= args.eval_after_epoch: # avoid unnecessary evaluation
            continue

        start_time = time.time()
        dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql')
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_acc'], best_result['iter'] = dev_acc, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
            logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))

    check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'))
    model.load_state_dict(check_point['model'])
    train_acc = decode('train', os.path.join(exp_path, 'train.iter' + str(best_result['iter'])), acc_type='sql')
    dev_acc_beam = decode('dev', output_path=os.path.join(exp_path, 'dev.iter' + str(best_result['iter']) + '.beam' + str(args.beam_size)), acc_type='beam')
    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc/Beam acc: %.4f/%.4f' % (best_result['iter'], best_result['dev_acc'], dev_acc_beam))
else:
    start_time = time.time()
    train_acc = decode('train', output_path=os.path.join(args.read_model_path, 'train.eval'), acc_type='sql')
    dev_acc = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval'), acc_type='sql')
    dev_acc_beam = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.beam' + str(args.beam_size)), acc_type='beam')
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match/inner beam acc is %.4f/%.4f ." % (time.time() - start_time, dev_acc, dev_acc_beam))
