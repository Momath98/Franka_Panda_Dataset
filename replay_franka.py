from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Paramètres
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--speed", type=float, default=1.0)
args = parser.parse_args()

EPISODE = args.episode
SPEED = args.speed

# Chemin du dataset Hugging Face
side_video_path = (
    PROJECT_DIR / "videos" / "side_camera"
    / f"episode_{EPISODE:06d}.mp4"
)

wrist_video_path = (
    PROJECT_DIR / "videos" / "wrist_camera"
    / f"episode_{EPISODE:06d}.mp4"
)




import os
# Charger les deux vidéos
side_cap = cv2.VideoCapture(side_video_path)
wrist_cap = cv2.VideoCapture(wrist_video_path)

# Vérifier leur ouverture
if not side_cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la vidéo Side Camera")

if not wrist_cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la vidéo Wrist Camera")

# Informations des vidéos
side_fps = side_cap.get(cv2.CAP_PROP_FPS)
wrist_fps = wrist_cap.get(cv2.CAP_PROP_FPS)

side_frames = int(side_cap.get(cv2.CAP_PROP_FRAME_COUNT))
wrist_frames = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Side Camera :", side_fps, "FPS,", side_frames, "frames")
print("Wrist Camera :", wrist_fps, "FPS,", wrist_frames, "frames")

print("Side camera :", os.path.exists(side_video_path))
print("Wrist camera :", os.path.exists(wrist_video_path))





if SPEED <= 0:
    raise ValueError("La vitesse doit être positive.")

# Chargement des données
df = pd.read_csv(PROJECT_DIR / "franka_all_episodes.csv")
df = df[df["episode_index"] == EPISODE].copy()
df = df.sort_values("frame_index").reset_index(drop=True)

if df.empty:
    raise ValueError(f"Épisode {EPISODE} introuvable.")

t = df["timestamp"].to_numpy()
n = len(df)

dt = np.median(np.diff(t)) if n > 1 else 0.1
interval = max(1, int(dt * 1000 / SPEED))

parameters = [
    ("position", "Positions articulaires", "Position (rad)"),
    ("velocity", "Vitesses articulaires", "Vitesse (rad/s)"),
    ("acceleration", "Accélérations articulaires", "Accélération (rad/s²)")
]

# Fenêtre graphique
# Fenêtre graphique
fig = plt.figure(figsize=(18, 16))
fig.suptitle(f"Franka Panda - Épisode {EPISODE}", fontsize=16)

# Organisation générale de la fenêtre
gs = fig.add_gridspec(
    4, 2,
    height_ratios=[2.2, 1.2, 1.2, 1.2],
    hspace=0.65,
    wspace=0.25,
    top=0.93,
    bottom=0.43
)
# Zones des vidéos
side_ax = fig.add_subplot(gs[0, 0])
wrist_ax = fig.add_subplot(gs[0, 1])

side_ax.set_title("Side Camera")
wrist_ax.set_title("Wrist Camera")

side_ax.axis("off")
wrist_ax.axis("off")

# Images initiales
side_ret, side_frame = side_cap.read()
wrist_ret, wrist_frame = wrist_cap.read()

if not side_ret or not wrist_ret:
    raise RuntimeError("Impossible de lire la première image des vidéos")

side_image = side_ax.imshow(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
wrist_image = wrist_ax.imshow(cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB))

# Zones des trois graphiques
axes = [
    fig.add_subplot(gs[1, :]),
    fig.add_subplot(gs[2, :]),
    fig.add_subplot(gs[3, :])
]

lines = []

for index, (ax, (parameter, title, ylabel)) in enumerate(zip(axes, parameters)):
    group = []

    for i in range(7):
        line, = ax.plot([], [], label=f"Joint {i}")
        group.append(line)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    ax.set_xlim(t[0], max(t[-1], t[0] + 0.01))
    ax.grid(True)
    ax.legend(loc="upper right", ncol=7, fontsize=8)

    values = np.concatenate([
        df[f"joint_{i}_{parameter}"].to_numpy()
        for i in range(7)
    ])

    ymin, ymax = np.nanmin(values), np.nanmax(values)
    padding = max((ymax - ymin) * 0.1, 0.01)
    ax.set_ylim(ymin - padding, ymax + padding)

    lines.append(group)

# Afficher le label du temps uniquement sous le dernier graphique
for ax in axes:
    ax.set_xlabel("")

fig.text(0.5, 0.225, "Temps (s)", ha="center", fontsize=10)
axes[-1].xaxis.set_label_position("bottom")

# Tableau numérique
table_ax = fig.add_axes([0.12, 0.245, 0.76, 0.15])
table_ax.axis("off")

def get_values(frame):
    row = df.iloc[frame]
    return [
        [
            f"{row[f'joint_{i}_position']:.4f}",
            f"{row[f'joint_{i}_velocity']:.4f}",
            f"{row[f'joint_{i}_acceleration']:.4f}"
        ]
        for i in range(7)
    ]

value_table = table_ax.table(
    cellText=get_values(0),
    rowLabels=[f"Joint {i}" for i in range(7)],
    colLabels=["Position (rad)", "Vitesse (rad/s)", "Accélération (rad/s²)"],
    cellLoc="center",
    loc="center"
)

value_table.auto_set_font_size(False)
value_table.set_fontsize(9)
value_table.scale(1, 1.4)

table_ax.set_title("Valeurs articulaires instantanées", pad=5)

# État de lecture
state = {
    "frame": 0,
    "playing": True,
    "updating_slider": False
}

time_text = fig.text(0.5, 0.205, "", ha="center", fontsize=11)

# Curseur temporel
slider_ax = fig.add_axes([0.15, 0.14, 0.70, 0.025])

time_slider = Slider(
    slider_ax,
    "Temps",
    t[0],
    max(t[-1], t[0] + 0.001),
    valinit=t[0]
)

# Dessin d'une image
def draw_frame(frame):
    frame = int(np.clip(frame, 0, n - 1))

    current_t = t[:frame + 1]

    # Mise à jour des graphiques
    for group, (parameter, _, _) in zip(lines, parameters):
        for i, line in enumerate(group):
            column = f"joint_{i}_{parameter}"
            line.set_data(
                current_t,
                df[column].to_numpy()[:frame + 1]
            )

    # Synchronisation des vidéos avec les données
    side_index = round(
        frame * (side_frames - 1) / max(n - 1, 1)
    )
    wrist_index = round(
        frame * (wrist_frames - 1) / max(n - 1, 1)
    )

    side_cap.set(cv2.CAP_PROP_POS_FRAMES, side_index)
    wrist_cap.set(cv2.CAP_PROP_POS_FRAMES, wrist_index)

    side_ok, side_frame = side_cap.read()
    wrist_ok, wrist_frame = wrist_cap.read()

    if side_ok:
        side_image.set_data(
            cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
        )

    if wrist_ok:
        wrist_image.set_data(
            cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
        )

    # Affichage du temps
    time_text.set_text(
        f"Épisode {EPISODE} | Temps : {t[frame]:.2f} s "
        f"| Image : {frame + 1}/{n}"
    )

    # Mise à jour du tableau
    new_values = get_values(frame)

    for row in range(7):
        for col in range(3):
            value_table[(row + 1, col)].get_text().set_text(
                new_values[row][col]
            )

    # Synchronisation du curseur
    state["updating_slider"] = True
    time_slider.set_val(t[frame])
    state["updating_slider"] = False

    return (
        [line for group in lines for line in group]
        + [time_text, side_image, wrist_image]
    )


def animate(_):
    if state["playing"]:
        draw_frame(state["frame"])

        if state["frame"] < n - 1:
            state["frame"] += 1
        else:
            state["playing"] = False

    return (
        [line for group in lines for line in group]
        + [time_text, side_image, wrist_image]
    )

# Gestion du curseur temporel
def slider_changed(value):
    if not state["updating_slider"]:
        frame = int(np.argmin(np.abs(t - value)))
        state["frame"] = frame
        draw_frame(frame)
        fig.canvas.draw_idle()


time_slider.on_changed(slider_changed)


# Bouton lecture / pause
button_ax = fig.add_axes([0.35, 0.075, 0.13, 0.04])
play_button = Button(button_ax, "Pause")


def toggle_play(event):
    state["playing"] = not state["playing"]
    play_button.label.set_text(
        "Pause" if state["playing"] else "Lecture"
    )


play_button.on_clicked(toggle_play)


# Réglage de la vitesse
speed_ax = fig.add_axes([0.58, 0.075, 0.25, 0.04])

speed_slider = Slider(
    speed_ax,
    "Vitesse",
    0.25,
    3.0,
    valinit=SPEED
)


def change_speed(value):
    animation.event_source.interval = max(
        1, int(1000 / side_fps / value)
    )


speed_slider.on_changed(change_speed)


# Création de l'animation
animation = FuncAnimation(
    fig,
    animate,
    frames=None,
    interval=max(1, int(1000 / side_fps / SPEED)),
    blit=False,
    cache_frame_data=False
)

# Affichage de la première image
draw_frame(0)

# Affichage de la fenêtre
plt.show()

# Libération des vidéos à la fermeture
side_cap.release()
wrist_cap.release()

