#coding=utf8
import os, math
import torch
import torch.nn as nn
from model.model_utils import rnn_wrapper, lens2mask, pad_single_seq_bert, PoolingFunction
from transformers import AutoModel, AutoConfig

class GraphInputLayer(nn.Module):

    def __init__(self, embed_size, hidden_size, word_vocab, dropout=0.2, fix_grad_idx=60, schema_aggregation='head+tail'):
        super(GraphInputLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_vocab = word_vocab
        self.fix_grad_idx = fix_grad_idx
        self.word_embed = nn.Embedding(self.word_vocab, self.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.embed_size, self.hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        self.word_embed.weight.grad[0].zero_() # padding symbol is always 0
        if index is not None:
            if not torch.is_tensor(index):
                index = torch.tensor(index, dtype=torch.long, device=self.word_embed.weight.grad.device)
            self.word_embed.weight.grad.index_fill_(0, index, 0.)
        else:
            self.word_embed.weight.grad[self.fix_grad_idx:].zero_()

    def forward(self, batch):
        question, table, column = self.word_embed(batch.questions), self.word_embed(batch.tables), self.word_embed(batch.columns)
        if batch.question_unk_mask is not None:
            question = question.masked_scatter_(batch.question_unk_mask.unsqueeze(-1), batch.question_unk_embeddings[:, :self.embed_size])
        if batch.table_unk_mask is not None:
            table = table.masked_scatter_(batch.table_unk_mask.unsqueeze(-1), batch.table_unk_embeddings[:, :self.embed_size])
        if batch.column_unk_mask is not None:
            column = column.masked_scatter_(batch.column_unk_mask.unsqueeze(-1), batch.column_unk_embeddings[:, :self.embed_size])
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs

class GraphInputLayerPLM(nn.Module):

    def __init__(self, plm='bert-base-uncased', hidden_size=256, dropout=0., subword_aggregation='mean',
            schema_aggregation='head+tail', lazy_load=False):
        super(GraphInputLayerPLM, self).__init__()
        self.plm_model = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join('./pretrained_models', plm))) \
            if lazy_load else AutoModel.from_pretrained(os.path.join('./pretrained_models', plm))
        self.config = self.plm_model.config
        self.subword_aggregation = SubwordAggregation(self.config.hidden_size, subword_aggregation=subword_aggregation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.config.hidden_size, hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        pass  

    def forward(self, batch):
        
        if batch.max_len <= 256:
            outputs = self.plm_model(**batch.inputs)[0] # final layer hidden states
            # batch_size x max_seq_len x hidden_size
        else:
            plm_outputs = []
            for idx, sample_id in enumerate(batch.inputs["input_ids"]):
                new_plm_input = {"input_ids": None, "attention_mask": None, "token_type_ids": None, "position_ids": None}
                if batch.input_lens[idx] <= 256:
                    new_plm_input["input_ids"] = sample_id[:256].unsqueeze(0)
                    new_plm_input["attention_mask"] = batch.inputs["attention_mask"][idx][:256].unsqueeze(0)
                    output = self.plm_model(**new_plm_input)[0]
                    # 1xseq_lenxhidden_size
                    output_size = output.size()
                    pad_tensor = output.new_zeros((output_size[0],
                                                   batch.max_len-output_size[1],
                                                   output_size[ -1]))
                    # pad to return to the original tensor size
                    output = torch.cat((output, pad_tensor), dim=1)
                    plm_outputs.append(output)
                else:
                    long_seq_output = self.encode_long_seq_input(batch, idx)
                    plm_outputs.append(long_seq_output)
            outputs = torch.cat(plm_outputs, dim=0)

        question, table, column = self.subword_aggregation(outputs, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs
    
    def encode_long_seq_input(self, batch, idx):
        sample = batch.examples[idx]
        question_ids = sample.question_id
        table_ids = sample.table_names_id
        column_ids = sample.column_names_id
        table_mask = lens2mask(torch.tensor(sample.table_subword_len).to(batch.device))
        col_mask = lens2mask(torch.tensor(sample.column_subword_len).to(batch.device))
        tabs = [pad_single_seq_bert(t, batch.tokenizer) for t in table_ids]
        cols = [pad_single_seq_bert(c, batch.tokenizer) for c in column_ids]
        question_output = self._bert_encode(question_ids, batch)
        # number of table x max table name len x hidden size
        table_output = self._bert_encode(tabs, batch)
        tabs = table_output.masked_select(table_mask.unsqueeze(-1))
        tabs = tabs.reshape((len(sample.table_id), self.config.hidden_size))
        col_output, last_sep = self._bert_en_code(cols, batch, True, sample)
        col_output = col_output.masked_select(col_mask.unsqueeze(-1))
        col_output = col_output.reshape((len(sample.column_id), self.config.hidden_size))
        col_output = torch.cat((col_output, last_sep.squeeze(1)), dim=0)
        final_output = torch.cat((question_output,
                                  table_output,
                                  col_output),
                                  dim=0)
        # pad if len of sample < max len of batch
        output_len = final_output.size(0)
        if output_len < batch.max_len:
            pad_tensor = torch.zeros((batch.max_len-output_len,
                                      self.config.hidden_size))
            final_output = torch.cat((final_output, pad_tensor), dim=0)
        return final_output
        
    # adapted from Ratsql
    def _bert_encode(self, toks, batch, is_cols=False, sample=None):
        if not isinstance(toks[0], list):  # encode question words
            tokens_tensor = torch.tensor([toks]).to(batch.device)
            outputs = self.plm_model(tokens_tensor)
            return outputs[0][0, :]  
        else:
            max_len = max([len(it) for it in toks])
            tok_ids = []
            for item_toks in toks:
                item_toks = item_toks + [batch.tokenizer.pad_token_id] * (max_len - len(item_toks))
                tok_ids.append(item_toks)

            tokens_tensor = torch.tensor(tok_ids).to(batch.device)
            outputs = self.plm_model(tokens_tensor)
            if is_cols and sample:
                # return the last encoded sep token 1x1x768
                last_sep_idx = len(sample.column_word_len[-1]) + 1 
                return outputs[0][:, 1: -1, :], outputs[0][-1:-1, last_sep_idx, :]
            return outputs[0][:, 1: -1, :] #remove cls and sep
        
         
class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, batch):
        """ Transform pretrained model outputs into our desired format
        questions: bsize x max_question_len x hidden_size
        tables: bsize x max_table_word_len x hidden_size
        columns: bsize x max_column_word_len x hidden_size
        """
        old_questions, old_tables, old_columns = inputs.masked_select(batch.question_mask_plm.unsqueeze(-1)), \
            inputs.masked_select(batch.table_mask_plm.unsqueeze(-1)), inputs.masked_select(batch.column_mask_plm.unsqueeze(-1))
        questions = old_questions.new_zeros(batch.question_subword_lens.size(0), batch.max_question_subword_len, self.hidden_size)
        questions = questions.masked_scatter_(batch.question_subword_mask.unsqueeze(-1), old_questions)
        tables = old_tables.new_zeros(batch.table_subword_lens.size(0), batch.max_table_subword_len, self.hidden_size)
        tables = tables.masked_scatter_(batch.table_subword_mask.unsqueeze(-1), old_tables)
        columns = old_columns.new_zeros(batch.column_subword_lens.size(0), batch.max_column_subword_len, self.hidden_size)
        columns = columns.masked_scatter_(batch.column_subword_mask.unsqueeze(-1), old_columns)

        questions = self.aggregation(questions, mask=batch.question_subword_mask)
        tables = self.aggregation(tables, mask=batch.table_subword_mask)
        columns = self.aggregation(columns, mask=batch.column_subword_mask)

        new_questions, new_tables, new_columns = questions.new_zeros(len(batch), batch.max_question_len, self.hidden_size),\
            tables.new_zeros(batch.table_word_mask.size(0), batch.max_table_word_len, self.hidden_size), \
                columns.new_zeros(batch.column_word_mask.size(0), batch.max_column_word_len, self.hidden_size)
        new_questions = new_questions.masked_scatter_(batch.question_mask.unsqueeze(-1), questions)
        new_tables = new_tables.masked_scatter_(batch.table_word_mask.unsqueeze(-1), tables)
        new_columns = new_columns.masked_scatter_(batch.column_word_mask.unsqueeze(-1), columns)
        return new_questions, new_tables, new_columns

class InputRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell='lstm', schema_aggregation='head+tail', share_lstm=False):
        super(InputRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell.upper()
        self.question_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_lstm = self.question_lstm if share_lstm else \
            getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_aggregation = schema_aggregation
        if self.schema_aggregation != 'head+tail':
            self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=schema_aggregation)

    def forward(self, input_dict, batch):
        """
            for question sentence, forward into a bidirectional LSTM to get contextual info and sequential dependence
            for schema phrase, extract representation for each phrase by concatenating head+tail vectors,
            batch.question_lens, batch.table_word_lens, batch.column_word_lens are used
        """
        questions, _ = rnn_wrapper(self.question_lstm, input_dict['question'], batch.question_lens, cell=self.cell)
        questions = questions.contiguous().view(-1, self.hidden_size)[lens2mask(batch.question_lens).contiguous().view(-1)]
        table_outputs, table_hiddens = rnn_wrapper(self.schema_lstm, input_dict['table'], batch.table_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            tables = self.aggregation(table_outputs, mask=batch.table_word_mask)
        else:
            table_hiddens = table_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else table_hiddens.transpose(0, 1)
            tables = table_hiddens.contiguous().view(-1, self.hidden_size)
        column_outputs, column_hiddens = rnn_wrapper(self.schema_lstm, input_dict['column'], batch.column_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            columns = self.aggregation(column_outputs, mask=batch.column_word_mask)
        else:
            column_hiddens = column_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else column_hiddens.transpose(0, 1)
            columns = column_hiddens.contiguous().view(-1, self.hidden_size)

        questions = questions.split(batch.question_lens.tolist(), dim=0)
        tables = tables.split(batch.table_lens.tolist(), dim=0)
        columns = columns.split(batch.column_lens.tolist(), dim=0)
        # dgl graph node feats format: q11 q12 ... t11 t12 ... c11 c12 ... q21 q22 ...
        outputs = [th for q_t_c in zip(questions, tables, columns) for th in q_t_c]
        outputs = torch.cat(outputs, dim=0)
        # transformer input format: bsize x max([q1 q2 ... t1 t2 ... c1 c2 ...]) x hidden_size
        # outputs = []
        # for q, t, c in zip(questions, tables, columns):
        #     zero_paddings = q.new_zeros((batch.max_len - q.size(0) - t.size(0) - c.size(0), q.size(1)))
        #     cur_outputs = torch.cat([q, t, c, zero_paddings], dim=0)
        #     outputs.append(cur_outputs)
        # outputs = torch.stack(outputs, dim=0) # bsize x max_len x hidden_size
        return outputs
