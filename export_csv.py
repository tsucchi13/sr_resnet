import os,sys
import pandas as pd
import glob


def main():
    dir_list = [n for n in os.listdir(".\\")\
                              if os.path.isdir(os.path.join(".\\",n))]
    for dir in dir_list:
        new_df = pd.DataFrame()
        dir_path = os.path.join(".\\", dir)
        print(dir_path)
        df = pd.read_csv(os.path.join(dir_path, "gaze_coords.csv"))
        for num in df.index:
            a = pd.Series([dir+"\\"+"cap_"+str(num)+".png"]).append(df.loc[num])

            new_df = new_df.append(a, ignore_index=True)
        new_df.to_csv(dir+"\\log.csv",index=False, header=False)

if __name__ == '__main__':
    main()
