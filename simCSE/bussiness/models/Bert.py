from torch import nn
from transformers import AlbertModel

class NeuralNetwork(nn.Module):
    def __init__(self,model_path, Config, output_way):
        super(NeuralNetwork, self).__init__()
        #self.bert = BertModel.from_pretrained(model_path,config=Config)
        self.bert = AlbertModel.from_pretrained(model_path,config=Config)
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:,0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output
