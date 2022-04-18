import numpy as np
import torch
import os
import time
import math
from models.actor_critic import Actor, Critic
from utils.rsmt_utils import Evaluator
from utils.log_utils import *
import argparse
from utils.plot_curves import plot_curve

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='exp', help='experiment name')
parser.add_argument('--degree', type=int, default=10, help='maximum degree of nets')
parser.add_argument('--dimension', type=int, default=2, help='terminal representation dimension')
parser.add_argument('--batch_size', type=int, default=256, help='test batch size')
parser.add_argument('--eval_size', type=int, default=10000, help='eval set size')
parser.add_argument('--num_batches', type=int, default=10000, help='number of batches')
parser.add_argument('--seed', type=int, default=9, help='random seed')

# Optimizer
parser.add_argument('--learning_rate', type=float, default=0.00005)
# parser.add_argument('--decay_rate', type=float, default=0.96)
# parser.add_argument('--decay_iter', type=int, default=5000)

# Hardcoded
log_intvl = 100

args = parser.parse_args()

device = torch.device("cuda:0")
# device = torch.device("cpu")

start_time = time.time()

print('experiment', args.experiment)
base_dir = 'save/'
exp_dir = base_dir + args.experiment + '/'
log_dir = exp_dir + 'rsmt' + str(args.degree) + '.log'
ckp_dir = exp_dir + 'rsmt' + str(args.degree) + '.pt'
best_ckp_dir = exp_dir + 'rsmt' + str(args.degree) + 'b.pt'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    print('Created exp_dir', exp_dir)
else:
    print('Exp_dir', exp_dir, 'already exists')
loger = LogIt(log_dir)

best_eval = 10.
best_kept = 0

actor = Actor(args.degree, device)
critic = Critic(args.degree, device)
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)
evaluator = Evaluator()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
eval_cases = np.random.rand(args.eval_size, args.degree, args.dimension)

start_batch = 1
if os.path.exists(ckp_dir):
    checkpoint = torch.load(ckp_dir)
    start_batch = checkpoint['batch_idx'] + 1
    print("Checkpoint exists. Continue training from batch", start_batch, ".")
    best_eval = checkpoint['best_eval']
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # change a seed so that different test cases will appear
    seed += start_batch % args.num_batches
    np.random.seed(seed)
    torch.manual_seed(seed)

for batch_idx in range(start_batch, start_batch + args.num_batches):
    actor.train()
    critic.train()
    input_batch = np.random.rand(args.batch_size, args.degree, args.dimension)
    outputs, log_probs = actor(input_batch)
    predictions = critic(input_batch)
    
    lengths = evaluator.eval_batch(input_batch, outputs.cpu().detach().numpy(), args.degree)
    length_tensor = torch.tensor(lengths, dtype=torch.float).to(device)

    with torch.no_grad():
        disadvantage = length_tensor - predictions
    actor_loss = torch.mean(disadvantage * log_probs)
    critic_loss = mse_loss(predictions, length_tensor)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
    optimizer.step()
    # if iter < args.decay_iter:
    #     scheduler.step()

    if batch_idx % log_intvl == 0:
        print('[batch', str(batch_idx) + ',', 'time', str(int(time.time() - start_time)) + 's]')
        print('length ', lengths.mean())
        print('predict', predictions.cpu().detach().numpy().mean())
        actor.eval()
        eval_lengths = []
        for eval_idx in range(math.ceil(args.eval_size / args.batch_size)):
            eval_batch = eval_cases[args.batch_size * eval_idx : args.batch_size * (eval_idx + 1)]
            with torch.no_grad():
                outputs, _ = actor(eval_batch, True)
            eval_lengths.append(evaluator.eval_batch(eval_batch, outputs.cpu().detach().numpy(), args.degree))
        eval_mean = np.concatenate(eval_lengths, -1).mean()
        if eval_mean < best_eval:
            best_eval = eval_mean
            best_kept = 0
            # keep a checkpoint anyway
            torch.save({
                'batch_idx': batch_idx,
                'best_eval': best_eval,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_ckp_dir)
            print('ckpt saved at', best_ckp_dir)
        else:
            best_kept += 1
        print('eval', eval_mean)
        print('best', best_eval, '(' + str(best_kept) + ')')
        
        print(outputs[0].cpu().detach().numpy().reshape(-1, 2).transpose(1, 0))

        loger.log_iter(batch_idx, {'eval' : eval_mean, 'best' : best_eval, 'time' : int(time.time() - start_time)})

torch.save({
    'batch_idx': batch_idx,
    'best_eval': best_eval,
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, ckp_dir)
print('ckpt saved at', ckp_dir)

plot_curve(log_dir)
