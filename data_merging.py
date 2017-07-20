import pandas

path = "./datasets/carlos/"
# up = pandas.read_csv(path+ "up_res.csv")
# down = pandas.read_csv(path+ "down_res.csv")
# left = pandas.read_csv(path+ "left_res.csv")
# right = pandas.read_csv(path+ "right_res.csv")
# center = pandas.read_csv(path+ "center_res.csv")

# def create_training_datasets(df_to_classify, others, filename):
#     n_samples = int(len(df_to_classify)/len(others))
#     # training_df = df_to_classify
#     training_df = df_to_classify.append(pandas.concat([df.sample(n=n_samples) for df in others]))
#     # for df in others:
#     #     training_df = training_df.append(df.sample(n=n_samples))
#     training_df.to_csv(path+filename)
#     return training_df

# up_training = create_training_datasets(up, [down, left, right, center], "up_training.csv")
# down_training = create_training_datasets(down, [up, left, right, center], "down_training.csv")
# left_training = create_training_datasets(left, [down, up, right, center], "left_training.csv")
# right_training = create_training_datasets(right, [down, left, up, center], "right_training.csv")
# center_training = create_training_datasets(center, [down, left, right, up], "center_training.csv")


csv_files = ["up3", "up4", "down3", "down4", "left3", "left4", "right3_1", "right3_2", "right4", "center3", "center4", "tv_right_3", "tv_left_3"]
csv_files = ["up1", "up2", "down1", "down2", "left1", "left2", "right1", "right2", "center1", "center2", "tv1", "tv2", "tv3"]

csvs = [pandas.read_csv(path+csv_file+"_res.csv") for csv_file in csv_files]
training_df = pandas.concat(csvs)
training_df.to_csv(path+"training_dataset_full.csv", index=False)
