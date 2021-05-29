from aa.config_utils import TrainingConfig
from aa.file_utils import * 
from tqdm import tqdm
import torch 
from torch import optim
#from aa.optimizers import Adam
LEARNING_RATE = 1e-05
# Creating the loss function and optimizer
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOSS = {"CrossEntropy": torch.nn.CrossEntropyLoss(), 
        "BCE": torch.nn.BCELoss(),
}


config_json= CONFIG_JSON["training_config"]
training_config = TrainingConfig._dict_from_json_file(config_json)


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def check_params_opt(default_params:dict,training_config:dict):
    for k,v in training_config.items(): 
        if k in default_params: 
            default_params[k] = v
    return default_params

def get_optimizer(model):
    """
    TODO: idem with other optimizers 
    """
    if training_config['optimizer'] =="Adam":
        opt = torch.optim.Adam(model.parameters())
        default_params = check_params_opt(opt.defaults,training_config)
        opt = torch.optim.Adam(model.parameters(), **default_params)
        return opt
    elif training_config['optimizer']=="Ada": 
        pass


class TrainingTransformer():    
    """
    TODO: define in transformer_config json or set default value?? 
    """
    def __init__(self,model):
        self.model = model
        self.epochs = training_config["epochs"] 
        self.loss_function = LOSS[training_config["loss"]]
        self.optimizer = get_optimizer(model)
        

    def train(self,training_loader):
        for epoch in range(self.epochs): 
            tr_loss = 0
            n_correct = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            self.model.train()
            for _,data in tqdm(enumerate(training_loader, 0)):
                input_ids = data['input_ids'].to(device, dtype = torch.long)
                attention_mask = data['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['labels'].to(device, dtype = torch.long)

                outputs = self.model(input_ids, 
                                attention_mask, 
                                token_type_ids
                                )
                loss = self.loss_function(outputs, targets)
                tr_loss += float(loss.item())
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += calcuate_accuracy(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples+=targets.size(0)
                
                if _%1000==0: # print every 1000 mini-batches
                    loss_step = tr_loss/nb_tr_steps
                    accu_step = (n_correct*100)/nb_tr_examples 
                    print(f"Training Loss per 1000 steps: {loss_step}")
                    print(f"Training Accuracy per 1000 steps: {accu_step}")

                self.optimizer.zero_grad()
                loss.backward()
                # # When using GPU
                self.optimizer.step()

            print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
            epoch_loss = tr_loss/nb_tr_steps
            epoch_accu = (n_correct*100)/nb_tr_examples
            print(f"Training Loss Epoch: {epoch_loss}")
            print(f"Training Accuracy Epoch: {epoch_accu}")
            return