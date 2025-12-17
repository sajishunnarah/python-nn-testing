import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create the dataset
# -------------------------------

data = {
    "Participant": ["Susan", "Chandler", "Jennifer", "Jason", "Sarah"],
    "Frustration (1-5)": [2, 2, 2, 3, 1],
    "Ease Before Leaving (1-5)": [3, 5, 5, 5, 4],
    "Ease While Driving (1-5)": [4, 3, 4, 3, 5],
    "Would Use (Yes/No)": ["Yes", "Yes", "Yes", "Yes", "Yes"]
}

df = pd.DataFrame(data)

# Convert Yes/No into numeric
df["Would Use (Binary)"] = df["Would Use (Yes/No)"].map({"Yes": 1, "No": 0})

# -------------------------------
# 2. Print descriptive statistics
# -------------------------------

print("\nDESCRIPTIVE STATISTICS\n")

print("Average frustration level:", df["Frustration (1-5)"].mean())
print("Average ease before leaving:", df["Ease Before Leaving (1-5)"].mean())
print("Average ease while driving:", df["Ease While Driving (1-5)"].mean())
print("Would realistically use (%):", df["Would Use (Binary)"].mean() * 100)

print("\nStandard Deviations:")
print("Frustration:", df["Frustration (1-5)"].std())
print("Ease before leaving:", df["Ease Before Leaving (1-5)"].std())
print("Ease while driving:", df["Ease While Driving (1-5)"].std())

# -------------------------------
# 3. Chart: Frustration
# -------------------------------

plt.figure()
plt.bar(df["Participant"], df["Frustration (1-5)"])
plt.title("Frustration Level by Participant")
plt.xlabel("Participant")
plt.ylabel("Frustration (1-5)")
plt.ylim(0, 5)
plt.show()

# -------------------------------
# 4. Chart: Ease Before Leaving
# -------------------------------

plt.figure()
plt.bar(df["Participant"], df["Ease Before Leaving (1-5)"])
plt.title("Ease of Use Before Leaving")
plt.xlabel("Participant")
plt.ylabel("Ease (1-5)")
plt.ylim(0, 5)
plt.show()

# -------------------------------
# 5. Chart: Ease While Driving
# -------------------------------

plt.figure()
plt.bar(df["Participant"], df["Ease While Driving (1-5)"])
plt.title("Ease of Use While Driving")
plt.xlabel("Participant")
plt.ylabel("Ease (1-5)")
plt.ylim(0, 5)
plt.show()

# -------------------------------
# 6. Optional: Export to CSV
# -------------------------------
# df.to_csv("parking_prototype_quantitative_data.csv", index=False)
