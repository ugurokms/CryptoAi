from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from tqdm import tqdm

class Sklearn_Classifier():
    def __init__(self, sklearn_classifier) -> None:
        self.model = sklearn_classifier

    def train_test(self, X_train, y_train, X_test, y_test):
        self.model = self.model.fit(X=X_train, y=y_train)
        y_pred = self.model.predict(X_test)
        f1_s = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        print(f"F1-score macro: {f1_s}")
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=False)/y_test.shape[0]
        print(f"Accuracy: {accuracy}")
        disp = ConfusionMatrixDisplay(confusion_matrix(y_true=y_test, y_pred=y_pred))
        disp.plot()

class ACNN(torch.nn.Module):
    """
    self attention + 1d cnn
    
    Taken from: https://github.com/hsd1503/resnet1d/tree/master
    @inproceedings{hong2020holmes,
                   title={HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units},
                   author={Hong, Shenda and Xu, Yanbo and Khare, Alind and Priambada, Satria and Maher, Kevin and Aljiffry, Alaa and Sun, Jimeng and Tumanov, Alexey},
                   booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
                   pages={1614--1624},
                   year={2020}}
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels, att_channels, n_len_seg, n_classes, verbose=False, save_best=False):
        super(ACNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_channels = att_channels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.save_best = save_best

        # (batch, channels, length)
        self.cnn = torch.nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=1, 
                            stride=4)

        self.W_att_channel = torch.nn.Parameter(torch.randn(self.out_channels, self.att_channels))
        self.v_att_channel = torch.nn.Parameter(torch.randn(self.att_channels, 1))

        self.dense = torch.nn.Linear(out_channels, n_classes)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        if self.verbose:
            print(out.shape)

        # (n_samples, n_channel, n_length) -> (n_samples, n_length, n_channel)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # (n_samples, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        if self.verbose:
            print(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # cnn
        out = self.cnn(out)
        if self.verbose:
            print(out.shape)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            print(out.shape)
        # global avg, (n_samples, n_seg, out_channels)
        out = out.view(-1, self.n_seg, self.out_channels)
        if self.verbose:
            print(out.shape)
        # self attention
        e = torch.matmul(out, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        out = torch.sum(torch.mul(gama, out), 1)
        if self.verbose:
            print(out.shape)
        # dense
        out = self.dense(out)
        if self.verbose:
            print(out.shape)
        
        return out
    
def train_test(train_dl, val_dl, model):
    # Hyperparameter
    lr = 0.001
    n_epochs = 100
    iterations_per_epoch = len(train_dl)
    best_acc = 0
    patience, trials = 500, 0
    base = 1
    step = 2
    loss_history = []
    acc_history = []
    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # train/eval
    print('Start model training')
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_dl):
            x, y = [t.to(device).to(torch.float) for t in batch]
            opt.zero_grad()
            out = model(x)
            y_gt = torch.tensor(y, dtype=torch.long)
            loss = criterion(out, y_gt)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
            
        epoch_loss /= iterations_per_epoch
        loss_history.append(epoch_loss)
        
        model.eval()
        correct, total = 0, 0
        for batch in val_dl:
            x, y = [t.to(device).to(torch.float) for t in batch]
            out = model(x)
            preds = torch.nn.functional.log_softmax(out, dim=1).argmax(dim=1)
            y_gt = torch.tensor(y, dtype=torch.long)
            total += y_gt.size(0)
            correct += (preds == y_gt).sum().item()
        
        acc = correct / total
        acc_history.append(acc)

        if epoch % base == 0:
            print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
            base *= step


        if acc > best_acc:
            trials = 0
            best_acc = acc
            if model.save_best:
                torch.save(model.state_dict(), 'best.pth')
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                print(f'Epoch {epoch} best model reached with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
                
    print('Done!')