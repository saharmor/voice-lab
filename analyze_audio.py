import sys
import opensmile
import pandas as pd

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python analyze_audio.py <audio_file_path>")
    #     sys.exit(1)

    # audio_file_path = sys.argv[1]

    # Initialize the openSMILE feature extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Extract features from the audio file
    features = smile.process_file('test.wav')

    # Print the extracted features
    pd.set_option('display.max_rows', None)
    print(features.T)

if __name__ == '__main__':
    main()
