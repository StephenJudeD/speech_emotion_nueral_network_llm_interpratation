import numpy as np
import matplotlib.pyplot as plt
import os

def plot_emotion_probabilities(emotion_probs):
    emotions = list(emotion_probs.keys())
    probabilities = [float(v.strip('%')) for v in emotion_probs.values()]

    # Create a Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(emotions, probabilities, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Probability (%)')
    plt.title('Emotion Probabilities')
    plt.ylim(0, 100)
    plt.grid(axis='y')

    # Save Bar Chart as an image in a static folder
    bar_chart_path = os.path.join('static', 'emotion_probabilities_bar.png')
    plt.savefig(bar_chart_path)
    plt.close()  # Close the figure

    # Create a Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    probabilities += probabilities[:1]  # Close the loop
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, probabilities, color='cyan', alpha=0.25)
    ax.plot(angles, probabilities, color='blue', linewidth=2)

    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions)
    ax.set_title('Emotion Probabilities Radar Chart')

    # Save Radar Chart as an image in a static folder
    radar_chart_path = os.path.join('static', 'emotion_probabilities_radar.png')
    plt.savefig(radar_chart_path)
    plt.close()  # Close the figure

    return bar_chart_path, radar_chart_path
