import pandas as pd
import matplotlib.pyplot as plt

def plot_burstgpt_line(csv_path="BurstGPT_processed.csv", start_epoch=0, end_epoch=96):
    df = pd.read_csv(csv_path)

    if start_epoch is not None and end_epoch is not None:
        df = df[(df['Epoch'] >= start_epoch) & (df['Epoch'] <= end_epoch)]

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Total Requests'], label='Total Requests', color='blue', linewidth=2)
    # plt.plot(df['Epoch'], df['Llama_8B Requests'], label='Llama_7B Requests', color='orange', linewidth=2)
    # plt.plot(df['Epoch'], df['Llama_70B Requests'], label='Llama_70B Requests', color='green', linewidth=2)

    plt.title("Requests vs. Epoch", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Number of Total Requests", fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('requests_vs_epoch.png', dpi=1200, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Total tokens'], label='Total tokens', color='blue', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_8B tokens'], label='ChatGPT tokens', color='orange', linewidth=2)
    plt.plot(df['Epoch'], df['Llama_70B tokens'], label='ChatGPT-4 tokens', color='green', linewidth=2)

    plt.title("Tokens vs. Epoch", fontsize=18)
    plt.xlabel("Epoch (15 minutes)", fontsize=18)
    plt.ylabel("Number of Tokens", fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tokens_vs_epoch.png', dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_burstgpt_line("BurstGPT_processed.csv")



