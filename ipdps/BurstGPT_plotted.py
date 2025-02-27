import pandas as pd
import matplotlib.pyplot as plt

def plot_burstgpt_line(csv_path="BurstGPT_processed.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Total Requests'], label='Total Requests', color='blue', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_8B Requests'], label='Llama_7B Requests', color='orange', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_70B Requests'], label='Llama_70B Requests', color='green', linewidth=2)

    plt.title("Requests vs. Epoch (Line Plot)")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Requests")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Total tokens'], label='Total tokens', color='blue', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_8B tokens'], label='Llama_7B tokens', color='orange', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_70B tokens'], label='Llama_70B tokens', color='green', linewidth=2)

    plt.title("Tokens vs. Epoch")
    plt.xlabel("Epoch (15 minutes)")
    plt.ylabel("Number of Tokens")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_burstgpt_line("BurstGPT_processed.csv")



