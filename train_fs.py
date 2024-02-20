import warnings

warnings.filterwarnings("ignore")
# from apex import amp
from torchvision import transforms
import torch
from torch import nn
import argparse
import model_fs
from dataset import RafDataSet
import kmeans_lloyd
import torch.nn.functional as F
from utils import utils
from visual2d import *
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/data/pzb/datasets/basic-RAFDB/', help='Raf-DB dataset path.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('-p2d', '--plot_2d', action='store_true')
    return parser.parse_args()


def run_training():
    args = parse_args()
    if args.wandb:
        import wandb
        wandb.init(project='raf-db')


    model = model_fs.ResNet18_Scale()
    # model = nn.DataParallel(model)
    model = model.cuda()
    # model = nn.DataParallel(model)
    # print(model)
    print("batch_size:", args.batch_size)



    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    params = model.parameters()
    args.lr = args.lr * (args.batch_size / 256)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
        if args.wandb:
            config = wandb.config
            config.learning_rate = args.lr
    else:
        raise ValueError("Optimizer not supported.")
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # model = model.cuda()
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    CE_criterion = torch.nn.CrossEntropyLoss()
    CS_criterion = nn.CosineSimilarity(dim=1).cuda()

    global plot


    best_acc = 0
    best_epoch = 0
    l1 = 0
    l2 = 0.1
    l3 = 0
    print("lambda:1=" + str(l1) + ",2=" + str(l2) + ",3=" + str(l3))
    all_loss = {"fm": [], "bs": [], "ce": [], "all": [], "test": []}
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        total = 0
        model.train()
        fm = 0.0
        bs = 0.0
        ce = 0.0
        all = 0.0

        plot = False
        if (i + 1) % 1 == 0 and args.plot_2d:
            plot = True
            all_features, all_labels = [], []

        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, cen_all, m2_all, m3_all, m1_all, x_out, out_feat = model(imgs)

            targets = targets.cuda()

            x4 = x_out[3]
            x3 = x_out[2]
            x2 = x_out[1]
            x1 = x_out[0]
            label, centroid = kmeans_lloyd.lloyd(x4.cuda(), 7, device=5, tol=1e-4)

            d20 = F.cosine_similarity(x2, centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d21 = F.cosine_similarity(x2, centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d22 = F.cosine_similarity(x2, centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d23 = F.cosine_similarity(x2, centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d24 = F.cosine_similarity(x2, centroid[4].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d25 = F.cosine_similarity(x2, centroid[5].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d26 = F.cosine_similarity(x2, centroid[6].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            # d4 = F.cosine_similarity(x2, centroid[4].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            d30 = F.cosine_similarity(x3, centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d31 = F.cosine_similarity(x3, centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d32 = F.cosine_similarity(x3, centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d33 = F.cosine_similarity(x3, centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d34 = F.cosine_similarity(x3, centroid[4].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d35 = F.cosine_similarity(x3, centroid[5].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d36 = F.cosine_similarity(x3, centroid[6].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            d10 = F.cosine_similarity(x1, centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d11 = F.cosine_similarity(x1, centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d12 = F.cosine_similarity(x1, centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d13 = F.cosine_similarity(x1, centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d14 = F.cosine_similarity(x1, centroid[4].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d15 = F.cosine_similarity(x1, centroid[5].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d16 = F.cosine_similarity(x1, centroid[6].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            data2_cs = torch.cat([d20, d21, d22, d23, d24, d25, d26], 0).t()
            _, label2_cs = torch.max(data2_cs, 1)

            data3_cs = torch.cat([d30, d31, d32, d33, d34, d35, d36], 0).t()
            _, label3_cs = torch.max(data3_cs, 1)

            data1_cs = torch.cat([d10, d11, d12, d13, d14, d15, d16], 0).t()
            _, label1_cs = torch.max(data1_cs, 1)

            m2 = []
            m3 = []
            m1 = []
            for j in range(7):
                if min(x2[(label2_cs == j)].shape) == 0:
                    m2.append(centroid[j])
                else:
                    m2.append(x2[(label2_cs == j)].mean(dim=0))
                if min(x3[(label3_cs == j)].shape) == 0:
                    m3.append(centroid[j])
                else:
                    m3.append(x3[(label3_cs == j)].mean(dim=0))
                if min(x1[(label1_cs == j)].shape) == 0:
                    m1.append(centroid[j])
                else:
                    m1.append(x1[(label1_cs == j)].mean(dim=0))

            layer2_mean = torch.stack(m2)
            layer3_mean = torch.stack(m3)
            layer1_mean = torch.stack(m1)
            CS12_loss = -CS_criterion(layer2_mean, centroid).mean() * 0.5
            CS13_loss = -CS_criterion(layer3_mean, centroid).mean() * 0.5
            CS11_loss = -CS_criterion(layer1_mean, centroid).mean() * 0.5

            CE_loss = CE_criterion(outputs, targets)

            CS2_loss = -CS_criterion(m2_all, cen_all).mean() * 0.5
            CS3_loss = -CS_criterion(m3_all, cen_all).mean() * 0.5
            CS1_loss = -CS_criterion(m1_all, cen_all).mean() * 0.5

            fm_loss = l2*CS2_loss + l1*CS1_loss + l3*CS3_loss
            bs_loss = l2*CS12_loss + l1*CS11_loss + l3*CS13_loss
            fm += fm_loss.__float__()
            bs += bs_loss.__float__()
            ce += CE_loss.__float__()

            loss = CE_loss + l2*CS2_loss + l1*CS1_loss + l3*CS3_loss + l2*CS12_loss + l1*CS11_loss + l3*CS13_loss

            # loss = CE_loss
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()


            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            total += targets.size(0)
            utils.progress_bar(batch_i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (train_loss / iter_cnt, 100. * correct_sum.float() / float(total),
                                  correct_sum, total))

        all_loss["fm"].append(fm / iter_cnt)
        all_loss["bs"].append(bs / iter_cnt)
        all_loss["ce"].append(ce / iter_cnt)

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt

        all_loss["all"].append(train_loss.__float__())

        # for param in model.parameters():
        #     print('{}:grad->{}'.format(param, param.grad))


        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
              (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
        # if plot:
        #     all_features = np.concatenate(all_features, 0)
        #     all_labels = np.concatenate(all_labels, 0)
        #     plot_features(all_features, all_labels, 7, i, './2d', 'RAF',
        #                   prefix='train' + str(i))
        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            if plot:
                all_features, all_labels = [], []
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                outputs, x4, x3, x2, x1, x_out, o_f = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)

                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()
                
                bingo_cnt += correct_or_not.sum().cpu()
                if plot:
                    if torch.cuda.is_available():
                        all_features.append(o_f.data.cpu().numpy())
                        all_labels.append(targets.data.cpu().numpy())
                    else:
                        all_features.append(o_f.data.numpy())
                        all_labels.append(targets.data.numpy())


            val_loss = val_loss / iter_cnt
            all_loss["test"].append(val_loss.__float__())
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, val_acc, val_loss))
            # print(all_loss)
            de = False
            if de:
                cm = confusion_matrix(gt_labels, pre_labels)
                cm = np.array(cm)
                # print(cm.shape)
                labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  # 横纵坐标标签
                # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
                plot_confusion_matrix(cm, labels_name, 'RAF-DB', str(val_acc)+"_"+str(i))
            if plot:
                all_features = np.concatenate(all_features, 0)
                all_labels = np.concatenate(all_labels, 0)
                plot_features(all_features, all_labels, 7, i, './2d', 'RAF',
                              prefix='test' + str(i))

            if args.wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )
            # if val_acc > 0.91 and val_acc > best_acc:
            if val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./remove0808/r6/fs', "epoch" + str(i) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = i
                print("best_acc:" + str(best_acc))
    print("lambda:1=" + str(l1) + ",2=" + str(l2) + ",3=" + str(l3))
    print("best_acc:" + str(best_acc))
    print("best_epoch:" + str(best_epoch))


if __name__ == "__main__":
    run_training()
