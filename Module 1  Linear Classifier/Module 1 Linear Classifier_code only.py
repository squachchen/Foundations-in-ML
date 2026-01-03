# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 08:48:18 2026

Do the following for your environment:
    pip install pandas numpy matplotlib datasets

@author: CHENPAUL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

df_load = load_dataset("scikit-learn/Fish")
df = df_load["train"].to_pandas()
#print(df)

# ============================================================
# Part 0.  Load data
# ============================================================
#df = pd.read_csv('Fish.csv') 
df['Volume'] = df['Height'] * (df['Length2']**2) * df['Width']

# Filter for Bream and Roach
df2 = df[df['Species'].isin(['Bream','Roach'])].copy()

# Separate species
bream = df2[df2['Species']=='Bream']
roach = df2[df2['Species']=='Roach']

# Extract x and y
x_b = bream['Volume'].values
y_b = bream['Weight'].values

x_r = roach['Volume'].values
y_r = roach['Weight'].values
plt.scatter(x_b, y_b, label='Bream')
plt.scatter(x_r, y_r, label='Roach')

plt.xlabel("Volume")
plt.ylabel("Weight")

plt.legend()
plt.show()

# ============================================================
# Part 1. Linear regression function
# ============================================================
def linear_regression(x, y):
    a = np.cov(x, y, bias=True)[0,1] / np.var(x)
    b = y.mean() - a * x.mean()
    return a, b

# Fit regressions
a_b, b_b = linear_regression(x_b, y_b)
a_r, b_r = linear_regression(x_r, y_r)

# Lines for plotting
x_line_b = np.linspace(x_b.min(), x_b.max(), 200)
y_line_b = a_b * x_line_b + b_b

x_line_r = np.linspace(x_r.min(), x_r.max(), 200)
y_line_r = a_r * x_line_r + b_r

# Plot
plt.figure(figsize=(9,6))
plt.scatter(x_b, y_b, label='Bream')
plt.scatter(x_r, y_r, label='Roach')

plt.plot(x_line_b, y_line_b, label='Bream Regression Line')
plt.plot(x_line_r, y_line_r, label='Roach Regression Line')

plt.xlabel("Volume")
plt.ylabel("Weight")
plt.title("Separate Linear Regression Lines for Bream and Roach")
plt.legend()
plt.show()

# ============================================================
# Part 2. Gradient Descent as a Linear Classifier
# ============================================================

# ------------------------------------------------------------
# STEP 1: BUILD THE DATASET (X) and LABELS (y)
# ------------------------------------------------------------
X = np.vstack([
    np.concatenate([x_b, x_r]),   # Volume
    np.concatenate([y_b, y_r])    # Weight
]).T

# Labels: +1 = Bream, -1 = Roach
y = np.concatenate([
    np.ones_like(x_b),
    -np.ones_like(x_r)
])

# ------------------------------------------------------------
# STEP 2: STANDARDIZE FEATURES (helps Gradient Descent)
# ------------------------------------------------------------
mu = X.mean(axis=0)
sigma = X.std(axis=0)
Xs = (X - mu) / sigma

# ------------------------------------------------------------
# STEP 3: INITIAL LINE (perpendicular to avg regression direction)
# ------------------------------------------------------------
a_ave = (a_b + a_r) / 2
w0_orig = np.array([1.0, a_ave])  # normal vector in ORIGINAL coords

mean_b = np.array([x_b.mean(), y_b.mean()])
mean_r = np.array([x_r.mean(), y_r.mean()])
midpoint = 0.5 * (mean_b + mean_r)

# Choose b so line passes through midpoint: w·midpoint + b = 0
b0_orig = -np.dot(w0_orig, midpoint)

# Convert boundary into STANDARDIZED coordinates
# score = w_std · x_std + b_std
w = w0_orig * sigma
b = np.dot(w0_orig, mu) + b0_orig

# -------- CRITICAL STABILITY FIX --------
# Scaling (w, b) by a constant DOES NOT change the boundary.
# But it DOES affect score magnitude and gradient stability.
# So we normalize to keep values reasonable.
def normalize_wb(w, b, eps=1e-12):
    s = np.linalg.norm(w)
    if s < eps:
        return w, b
    return w / s, b / s

w, b = normalize_wb(w, b)

# ------------------------------------------------------------
# STEP 4: Helper functions
# ------------------------------------------------------------

def sigmoid_stable(z):
    """
    Stable sigmoid:
      sigmoid(z) = 1 / (1 + exp(-z))
    This version avoids overflow by clipping.
    """
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(y, scores):
    """
    Numerically stable logistic loss:
      L = mean( log(1 + exp(-y*score)) )
    uses logaddexp to avoid overflow.
    """
    margins = y * scores
    return np.mean(np.logaddexp(0.0, -margins))

def plot_boundary_segment(ax, w1, w2, b, x_min, x_max, y_min, y_max, label, linestyle='-'):
    """
    Draw the segment of the line w1*x + w2*y + b = 0 that lies within
    the rectangle [x_min,x_max] x [y_min,y_max].

    This is MUCH more reliable than sampling & clipping.
    """
    pts = []

    # Intersections with left/right edges (x = const)
    if abs(w2) > 1e-12:
        y_at_xmin = -(w1 * x_min + b) / w2
        y_at_xmax = -(w1 * x_max + b) / w2
        if y_min <= y_at_xmin <= y_max:
            pts.append((x_min, y_at_xmin))
        if y_min <= y_at_xmax <= y_max:
            pts.append((x_max, y_at_xmax))

    # Intersections with bottom/top edges (y = const)
    if abs(w1) > 1e-12:
        x_at_ymin = -(w2 * y_min + b) / w1
        x_at_ymax = -(w2 * y_max + b) / w1
        if x_min <= x_at_ymin <= x_max:
            pts.append((x_at_ymin, y_min))
        if x_min <= x_at_ymax <= x_max:
            pts.append((x_at_ymax, y_max))

    # If we found at least 2 intersection points, draw segment between them
    if len(pts) >= 2:
        # pick two farthest points (more robust if we got 3-4 points)
        best = (pts[0], pts[1])
        best_d = -1
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = dx*dx + dy*dy
                if d > best_d:
                    best_d = d
                    best = (pts[i], pts[j])

        (x1, y1), (x2, y2) = best
        ax.plot([x1, x2], [y1, y2], linestyle=linestyle, label=label)

    # If no intersections, the line is completely outside the visible box.
    # In that case we draw nothing (but your axis won't get wrecked).

def wait_for_space(fig):
    """
    Blocks until SPACE is pressed. Press 'q' to quit.
    """
    state = {"quit": False}

    def on_key(event):
        if event.key == " ":
            fig.canvas.stop_event_loop()
        elif event.key == "q":
            state["quit"] = True
            fig.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.start_event_loop(timeout=-1)
    fig.canvas.mpl_disconnect(cid)

    if state["quit"]:
        raise SystemExit("Quit pressed (q). Stopping training.")

# ------------------------------------------------------------
# STEP 5: Plot initial boundary (Original coordinates)
# ------------------------------------------------------------

# Axis limits based ONLY on data (so we always see the fish properly)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_pad = 0.05 * (x_max - x_min)
y_pad = 0.10 * (y_max - y_min)
x_min -= x_pad; x_max += x_pad
y_min -= y_pad; y_max += y_pad

# Convert boundary back to ORIGINAL for plotting:
# w_orig = w_std / sigma
# b_orig = b_std - w_orig·mu
w_orig = w / sigma
b_orig = b - np.dot(w_orig, mu)

plt.figure(figsize=(9, 6))
plt.scatter(x_b, y_b, label="Bream")
plt.scatter(x_r, y_r, label="Roach")

plot_boundary_segment(
    plt.gca(),
    w_orig[0], w_orig[1], b_orig,
    x_min, x_max, y_min, y_max,
    label="Initial Boundary", linestyle="--"
)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Volume")
plt.ylabel("Weight")
plt.title("Part 2: Start Line (Before Gradient Descent)")
plt.legend()
plt.show()

# ------------------------------------------------------------
# STEP 6: Gradient Descent + step-by-step visualization
# ------------------------------------------------------------

lr = 0.2       # learning rate (try 0.05, 0.1, 0.2)
epochs = 50

# Standardized plot limits
z1_min, z1_max = Xs[:, 0].min(), Xs[:, 0].max()
z2_min, z2_max = Xs[:, 1].min(), Xs[:, 1].max()
z1_pad = 0.05 * (z1_max - z1_min)
z2_pad = 0.05 * (z2_max - z2_min)
z1_min -= z1_pad; z1_max += z1_pad
z2_min -= z2_pad; z2_max += z2_pad

# Precompute standardized class points for plotting
Xb_s = (np.vstack([x_b, y_b]).T - mu) / sigma
Xr_s = (np.vstack([x_r, y_r]).T - mu) / sigma

fig, (ax_orig, ax_std) = plt.subplots(1, 2, figsize=(14, 6))

for epoch in range(1, epochs + 1):

    # ----- forward -----
    scores = Xs @ w + b
    loss = logistic_loss(y, scores)

    # ----- gradients -----
    margins = y * scores
    p = sigmoid_stable(-margins)

    grad_w = np.mean((-y[:, None] * Xs) * p[:, None], axis=0)
    grad_b = np.mean((-y) * p)

    # ----- update -----
    w -= lr * grad_w
    b -= lr * grad_b

    # Optional but helpful: keep (w,b) normalized so scores don't explode later
    w, b = normalize_wb(w, b)

    # ----- redraw -----
    ax_orig.cla()
    ax_std.cla()

    # LEFT: original scale
    ax_orig.scatter(x_b, y_b, label="Bream")
    ax_orig.scatter(x_r, y_r, label="Roach")

    w_orig = w / sigma
    b_orig = b - np.dot(w_orig, mu)

    plot_boundary_segment(
        ax_orig,
        w_orig[0], w_orig[1], b_orig,
        x_min, x_max, y_min, y_max,
        label=f"Boundary (epoch {epoch})"
    )

    ax_orig.set_xlim(x_min, x_max)
    ax_orig.set_ylim(y_min, y_max)
    ax_orig.set_xlabel("Volume")
    ax_orig.set_ylabel("Weight")
    ax_orig.set_title("Original Scale (Volume vs Weight)")
    ax_orig.legend()

    # RIGHT: standardized scale
    ax_std.scatter(Xb_s[:, 0], Xb_s[:, 1], label="Bream (std)")
    ax_std.scatter(Xr_s[:, 0], Xr_s[:, 1], label="Roach (std)")

    plot_boundary_segment(
        ax_std,
        w[0], w[1], b,
        z1_min, z1_max, z2_min, z2_max,
        label=f"Boundary (epoch {epoch})"
    )

    ax_std.set_xlim(z1_min, z1_max)
    ax_std.set_ylim(z2_min, z2_max)
    ax_std.set_xlabel("z-Volume")
    ax_std.set_ylabel("z-Weight")
    ax_std.set_title("Standardized Scale (what GD trains on)")
    ax_std.legend()

    # Show numeric info
    fig.suptitle(
        f"Epoch {epoch}/{epochs} | lr={lr} | loss={loss:.4f}\n"
        f"grad_w=[{grad_w[0]:.4f}, {grad_w[1]:.4f}] | grad_b={grad_b:.4f}   "
        f"(SPACE=next, q=quit)",
        fontsize=11
    )

    fig.canvas.draw_idle()
    plt.show(block=False)

    wait_for_space(fig)

plt.show()

# ------------------------------------------------------------
# STEP 7 Save the final weights and bias
# ------------------------------------------------------------
w_final = w.copy()
b_final = b

# ============================================================
# Part 3. User Input → Prediction (Final Epoch Model)
# ============================================================

# ------------------------------------------------------------
# STEP 1: Valid input ranges (from training data)
# ------------------------------------------------------------
length_min, length_max = df2['Length2'].min(), df2['Length2'].max()
width_min,  width_max  = df2['Width'].min(),  df2['Width'].max()
height_min, height_max = df2['Height'].min(), df2['Height'].max()
weight_min, weight_max = df2['Weight'].min(), df2['Weight'].max()

print("Enter values within these ranges:")
print(f"  Length : [{length_min:.1f}, {length_max:.1f}]")
print(f"  Width  : [{width_min:.1f},  {width_max:.1f}]")
print(f"  Height : [{height_min:.1f}, {height_max:.1f}]")
print(f"  Weight : [{weight_min:.1f}, {weight_max:.1f}]")
print("Good test values are Bream 500 30 14 5. Roach is 110 20 6 3 ")
# ------------------------------------------------------------
# STEP 2: User input
# ------------------------------------------------------------
weight = float(input("Weight: "))
length = float(input("Length: "))
height = float(input("Height: "))
width  = float(input("Width : "))


# Optional: basic range check
assert length_min <= length <= length_max, "Length out of range"
assert width_min  <= width  <= width_max,  "Width out of range"
assert height_min <= height <= height_max, "Height out of range"
assert weight_min <= weight <= weight_max, "Weight out of range"

# ------------------------------------------------------------
# STEP 3: Feature construction (must match training)
# ------------------------------------------------------------
volume = height * (length ** 2) * width
x_user = np.array([volume, weight])

# Standardize using TRAINING statistics
x_user_std = (x_user - mu) / sigma

# ------------------------------------------------------------
# STEP 4: Prediction using FINAL model
# ------------------------------------------------------------
score = np.dot(w_final, x_user_std) + b_final

prediction = "Bream" if score >= 0 else "Roach"
confidence = 1.0 / (1.0 + np.exp(-abs(score)))  # optional, intuitive

# ------------------------------------------------------------
# STEP 5: Output
# ------------------------------------------------------------
print("\n--- Prediction (Final Epoch Model) ---")
print(f"Volume computed : {volume:.2f}")
print(f"Raw score       : {score:.4f}")
print(f"Prediction      : {prediction}")
print(f"Confidence*     : {confidence:.2f}")

print("\n*Confidence is derived from distance to the decision boundary.")

plt.figure(figsize=(7, 5))
plt.scatter(x_b, y_b, label="Bream")
plt.scatter(x_r, y_r, label="Roach")

plt.scatter(volume, weight,
            s=150,
            marker="*",
            color="black",
            label="Your Fish")

# Final decision boundary
w_orig = w_final / sigma
b_orig = b_final - np.dot(w_orig, mu)

plot_boundary_segment(
    plt.gca(),
    w_orig[0], w_orig[1], b_orig,
    x_min, x_max, y_min, y_max,
    label="Final boundary"
)

plt.xlabel("Volume")
plt.ylabel("Weight")
plt.title("Final Model Prediction (Epoch 50)")
plt.legend()
plt.show()
# ------------------------------------------------------------
# STEP 7 Save the final weights and bias
# ------------------------------------------------------------
w_final = w.copy()
b_final = b

# ============================================================
# Part 3. User Input → Prediction (Final Epoch Model)
# ============================================================

# ------------------------------------------------------------
# STEP 1: Valid input ranges (from training data)
# ------------------------------------------------------------
length_min, length_max = df2['Length2'].min(), df2['Length2'].max()
width_min,  width_max  = df2['Width'].min(),  df2['Width'].max()
height_min, height_max = df2['Height'].min(), df2['Height'].max()
weight_min, weight_max = df2['Weight'].min(), df2['Weight'].max()

print("Enter values within these ranges:")
print(f"  Length : [{length_min:.1f}, {length_max:.1f}]")
print(f"  Width  : [{width_min:.1f},  {width_max:.1f}]")
print(f"  Height : [{height_min:.1f}, {height_max:.1f}]")
print(f"  Weight : [{weight_min:.1f}, {weight_max:.1f}]")
print("Good test values are Bream 500 30 14 5. Roach is 110 20 6 3 ")
# ------------------------------------------------------------
# STEP 2: User input
# ------------------------------------------------------------
weight = float(input("Weight: "))
length = float(input("Length: "))
height = float(input("Height: "))
width  = float(input("Width : "))


# Optional: basic range check
assert length_min <= length <= length_max, "Length out of range"
assert width_min  <= width  <= width_max,  "Width out of range"
assert height_min <= height <= height_max, "Height out of range"
assert weight_min <= weight <= weight_max, "Weight out of range"

# ------------------------------------------------------------
# STEP 3: Feature construction (must match training)
# ------------------------------------------------------------
volume = height * (length ** 2) * width
x_user = np.array([volume, weight])

# Standardize using TRAINING statistics
x_user_std = (x_user - mu) / sigma

# ------------------------------------------------------------
# STEP 4: Prediction using FINAL model
# ------------------------------------------------------------
score = np.dot(w_final, x_user_std) + b_final

prediction = "Bream" if score >= 0 else "Roach"
confidence = 1.0 / (1.0 + np.exp(-abs(score)))  # optional, intuitive

# ------------------------------------------------------------
# STEP 5: Output
# ------------------------------------------------------------
print("\n--- Prediction (Final Epoch Model) ---")
print(f"Volume computed : {volume:.2f}")
print(f"Raw score       : {score:.4f}")
print(f"Prediction      : {prediction}")
print(f"Confidence*     : {confidence:.2f}")

print("\n*Confidence is derived from distance to the decision boundary.")

plt.figure(figsize=(7, 5))
plt.scatter(x_b, y_b, label="Bream")
plt.scatter(x_r, y_r, label="Roach")

plt.scatter(volume, weight,
            s=150,
            marker="*",
            color="black",
            label="Your Fish")

# Final decision boundary
w_orig = w_final / sigma
b_orig = b_final - np.dot(w_orig, mu)

plot_boundary_segment(
    plt.gca(),
    w_orig[0], w_orig[1], b_orig,
    x_min, x_max, y_min, y_max,
    label="Final boundary"
)

plt.xlabel("Volume")
plt.ylabel("Weight")
plt.title("Final Model Prediction (Epoch 50)")
plt.legend()
plt.show()
