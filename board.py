from torch.utils.tensorboard import SummaryWriter  

writer = SummaryWriter('./output/output1')
step = 0
with open("output/train_generation_simple/2023-02-27-22-27-51/output.log") as f:
    for line in f.readlines():
        p = line.find("diff_loss")
        if p != -1:
            writer.add_scalar('diff_loss', float(line[p+15:p+21]), step)
            step = step + 1
