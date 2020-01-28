import yaml
import pathlib
import pandas as pd
import addict

cfg = """
log_dir: $LOG/set_rec/2018-12-29/simulation_emb_size_grid_large_hidden
"""

if __name__ == '__main__':
  cfg = adddict.Dict(yaml.load(cfg))

  for i, directory in enumerate(cfg.log_dir.glob('*')):
    name = directory.parts[-1]
    dir_dict = dict(name=name)
    for csv in directory.glob('*.csv'):
      with csv.open('r') as f:
        step, value = f.readlines()[-1].split(',')
        dir_dict[csv.name.rstrip('.csv')] = value.rstrip()
        dir_dict['step'] = step
    if i == 0:
      df = pd.DataFrame({}, columns=list(dir_dict.keys()))
    df = df.append(dir_dict, ignore_index=True)
 
  df.to_csv(cfg.log_dir / 'results.csv')
  df = df.dropna()
  df_filter = pd.DataFrame({}, columns=list(df.columns) + ['emb_size', 'model'])
  for model in ['Deep', 'InnerProduct', 'ResidualInnerProduct']:
    df_model = df[df.name.str.contains('model=%s' % model)] 
    for emb_size in [8, 9, 16, 32, 64, 128]:
      df_ = df_model[df_model.name.str.contains('emb_size=%d' % emb_size)]
      df_ = df_.sort_values(by='valid_in_matrix_recall', ascending=False)
      df_['emb_size'] = emb_size
      df_['model'] = model
      df_filter = df_filter.append(df_.head(1), ignore_index=True)

  df_filter.to_csv(cfg.log_dir / 'sorted_by_valid_in_matrix_recall.csv')

  


      
      
    
