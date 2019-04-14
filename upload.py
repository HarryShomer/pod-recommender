import pandas as pd
import json
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def fix_string(s):
	return s.encode("ascii", "ignore").decode("utf-8")


def fix_unicode(model):
	with open(os.path.join(FILE_PATH, f"{model}_6_results.json"), "r") as f:
		results = json.load(f)

	for i in range(len(results['results'])):
		results['results'][i]["Podcast"] = fix_string(results['results'][i]["Podcast"])
		results['results'][i]["Episode"] = fix_string(results['results'][i]["Episode"])

		for j in range(1, 6):
			if f"Recommendation #{j}" in results['results'][i]:
				results['results'][i][f"Recommendation #{j}"]['podcast'] = fix_string(results['results'][i][f"Recommendation #{j}"]['podcast'])
				results['results'][i][f"Recommendation #{j}"]['episode'] = fix_string(results['results'][i][f"Recommendation #{j}"]['episode'])

	with open(f"{model}_6_results.json", "w") as f:
		json.dump(results, f, indent=4)


################################################
################################################
################################################
fix_unicode("lda")
fix_unicode("lsi")
"""
with open(os.path.join(FILE_PATH, "lda_6_results.json"), "r") as f:
		results = json.load(f)


recs = []
for rec in results['results']:
	episode = {'podcast': rec['Podcast'], "episode": rec['Episode']}
	for i in range(1, 6):
		if f"Recommendation #{i}" in rec:
			episode[f'episode_rec_{i}'] = rec[f"Recommendation #{i}"]['episode']
			episode[f'podcast_rec_{i}'] = rec[f"Recommendation #{i}"]['podcast']

	recs.append(episode)

df = pd.DataFrame(recs)
print(df.iloc[0])
"""


