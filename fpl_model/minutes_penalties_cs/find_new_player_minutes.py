import pandas as pd
import numpy as np
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"

new_players_list = pd.read_csv(_DATA / 'new_players_2026.csv')
player_match_stats = pd.read_csv(_DATA / 'player_match_stats.csv', encoding='utf-8-sig')

all_players = np.unique(player_match_stats['Player']) # numpy array
new_players = np.unique(new_players_list['Players']) # numpy array
print(len(new_players))

out_df = pd.DataFrame(columns=['Player', 'Mins'])
for player in new_players:
    if player in all_players:
        player_mins = player_match_stats.loc[player_match_stats['Player'] == player, 'Minutes'].sum()
        out_df.loc[len(out_df)] = [player, player_mins]
    else:
        out_df.loc[len(out_df)] = [player, 0]


out_df.to_csv(_DATA / 'new_players_2026.csv', index=False, encoding="utf-8-sig")