import copy
import jobs
import collections
import pathlib


def get_slurm_script_gpu(train_dir, command, time):
  """Returns contents of SLURM script for a gpu job."""
  return """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:tesla_p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --output={}/slurm_%j.out
#SBATCH -t {}
module load anaconda3 cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1
source activate yumi
{}
""".format(train_dir, time, command)


if __name__ == '__main__':
  commands = ["PYTHONPATH=. python3 experiment/arxiv/torch_rank_from_sets.py"]

  ## global options
  grid = collections.OrderedDict()
  grid['log_dir'] = pathlib.Path(pathlib.os.environ['TMPDIR']) / 'set_rec' #/ 'arxiv'
  grid['experiment_name'] = 'log_interval=500_batch=512_1024'

  grid['num_workers'] = 8
  grid['pin_memory'] = True
  grid['max_iterations'] = [100000]
# grid['number_of_recommendations'] = ["10 30 50 70 100"]
  grid['number_of_recommendations'] = [100]
  grid['use_gpu'] = [True]

  grid = copy.deepcopy(grid)
  grid['model'] = ['InnerProduct']
  grid['emb_size'] = [9]
  grid['momentum'] = [0.9]
  grid['learning_rate'] = [0.1]
  grid['dropout'] = [0.0]
  grid['batch_size'] = [512]
  grid['log_interval'] = 500
  grid['time'] = '05:59:00'
  grid['lr_decay'] = ['linear']
  keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
  keys_for_dir_name.insert(0, 'batch_size')
  keys_for_dir_name.insert(0, 'emb_size')
  keys_for_dir_name.insert(0, 'model')

  for cfg in jobs.param_grid(grid):
    cfg['train_dir'] = jobs.make_train_dir(cfg, keys_for_dir_name)
    jobs.submit(commands, cfg, get_slurm_script_gpu)

  grid = copy.deepcopy(grid)
  grid['model'] = ['Deep', 'ResidualInnerProduct']
  grid['emb_size'] = 8
#  grid['eval_batch_size'] = 1024
  grid['hidden_size'] = 128

  # for cfg in jobs.param_grid(grid):
  #   cfg['train_dir'] = jobs.make_train_dir(cfg, keys_for_dir_name)
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)
  

  def launch_grid(grid):
    grid = copy.deepcopy(grid)
    grid['model'] = 'InnerProduct'
    grid['lr_decay'] = 'linear'
    grid['emb_size'] = 9
    for cfg in jobs.param_grid(grid):
      cfg['train_dir'] = jobs.make_train_dir(cfg, keys_for_dir_name)
      jobs.submit(commands, cfg, get_slurm_script_gpu)

    grid = copy.deepcopy(grid)
    grid['model'] = ['Deep', 'ResidualInnerProduct']
    grid['emb_size'] = 8
    grid['eval_batch_size'] = 1024
    grid['lr_decay'] = 'plateau'
    grid['hidden_size'] = 128
    grid['learning_rate'] = 0.01
    for cfg in jobs.param_grid(grid):
      cfg['train_dir'] = jobs.make_train_dir(cfg, keys_for_dir_name)
      jobs.submit(commands, cfg, get_slurm_script_gpu)
      
  for i in range(30):
    grid['experiment_name'] = 's_%d' % i
    grid['train_tsv'] = '/tmp/dat/set_rec/simulation_%d/train.tsv' % i
    grid['valid_tsv'] = '/tmp/dat/set_rec/simulation_%d/valid.tsv' % i
    grid['test_tsv'] = '/tmp/dat/set_rec/simulation_%d/test.tsv' % i
    grid['item_attributes'] = '/tmp/dat/set_rec/simulation_%d/item_attributes_csr.npz' % i
    grid['test_users_tsv'] = '/tmp/dat/set_rec/simulation_%d/test_users.tsv' % i

    launch_grid(grid)
