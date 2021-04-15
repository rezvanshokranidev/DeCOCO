import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import EmoryNLPRobertaCometDataset
from model import MaskedNLLLoss
from commonsense_model import CommonsenseGRUModel
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path as path


def create_class_weight(mu=1):
    #Smoothen Weights Technique, The log function smooths the weights for the imbalanced class.
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 2099, 1: 968, 2: 831, 3: 3095, 4: 595, 5: 717, 6: 1184}
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_EmoryNLP_loaders(batch_size=32, num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset('train')
    validset = EmoryNLPRobertaCometDataset('valid')
    testset = EmoryNLPRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        log_prob = model(r, x5, x6, x1, o2, o3, qmask, umask)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, [avg_fscore]

if __name__ == '__main__':
    import gc
    torch.cuda.empty_cache()
    gc.collect()


    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--recurrent_dropout',
                        type=float,
                        default=0.3,
                        metavar='recurrent_dropout',
                        help='recurrent_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general2', help='Attention type in context GRU')
    parser.add_argument('--seed', type=int, default=500, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=0, help='normalization strategy')
    parser.add_argument('--mu', type=float, default=1, help='class_weight_mu')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')


    emo_gru = True
    n_classes = 7
    '''sad, mad, scared, powerful, peaceful, joyful, and neutral
    '''
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100
    D_ca = 100

    D_e = D_p + D_r + D_i

    global seed
    seed = args.seed
    # seed_everything(seed)

    model = CommonsenseGRUModel(D_m,
                                D_s,
                                D_g,
                                D_p,
                                D_r,
                                D_i,
                                D_e,
                                D_h,
                                D_a,
                                D_ca,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                recurrent_dropout=args.recurrent_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                norm=args.norm)

    print ('EmoryNLP DeCOCO Model.')

    if cuda:
        model.cuda()


    if args.class_weight:
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:
            loss_weights = torch.FloatTensor([0.5, 1, 1, 0.3, 1, 1, 0.9])
            # counts {0: 2099, 1: 968, 2: 831, 3: 3095, 4: 595, 5: 717, 6: 1184}

        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    dir = path.cwd()
    lf = open(dir / 'train/logs/cosmic_emory_emotion_logs.txt', 'a')


    train_loader, valid_loader, test_loader = get_EmoryNLP_loaders(batch_size=batch_size,
                                                                   num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        valid_loss, valid_acc, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)

        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)

        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))

        print (x)
        lf.write(x + '\n')


    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]

    scores = [score1, score2]
    scores_val_loss = [score1]
    scores_val_f1 = [score2]
    scores = [str(item) for item in scores]

    print ('Test Scores: Micro w/o Neutral, Macro')
    print('F1@Best Valid Loss: {}'.format(scores_val_loss))
    print('F1@Best Valid F1: {}'.format(scores_val_f1))


    rf = open(dir / 'train/results/cosmic_emorynlp_emotion_results.txt', 'a')


    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()
