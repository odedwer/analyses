
sampled_Tmean = np.mean(data["Amplitude"])
sampled_meansRows = [np.mean(data["Amplitude"][data["Location"]=="Yes"]),
                     np.mean(data["Amplitude"][data["Location"]=="No"])]
sampled_meansCols = [np.mean(data["Amplitude"][data["Coffee"]==1]),
                     np.mean(data["Amplitude"][data["Coffee"]==0])]
sampled_means = [np.mean(data["Amplitude"][(data["Location"]=="Yes") & (data["Coffee"]==1)]),
                 np.mean(data["Amplitude"][(data["Location"]=="No")  & (data["Coffee"]==1)]),
                 np.mean(data["Amplitude"][(data["Location"]=="Yes") & (data["Coffee"]==0)]),
                 np.mean(data["Amplitude"][(data["Location"]=="No")  & (data["Coffee"]==0)])]

print("Mean of all groups together (X) = ", round(sampled_Tmean,3))
from tabulate import tabulate
print(tabulate([['Yes (X_1.)', sampled_means[0],sampled_means[2],sampled_meansRows[0]],
                ['No (X_2.)', sampled_means[1],sampled_means[3],sampled_meansRows[1]],
                ['Mean', sampled_meansCols[0],sampled_meansCols[1],sampled_Tmean] ],
               headers=['Location', 'Social science ( X_.1 )', "Forum ( X_.2 )", "Humanities ( X_.3 )", "Mean"]))

print("\n\n",np.round(aov_table,3))

print("\n\n**SSinteraction** = sum of (Xij -Xi. - X.j - X)^2 = ",round(aov_table.sum_sq[2],3))
print("F = MSInteraction/MSWithin: (",round(aov_table.sum_sq[2],3),
      "/",round(aov_table.df[2],3),")/ (",
      round(aov_table.sum_sq[3],3),"/",round(aov_table.df[3],3),") =",round(Fint,3))
print("p-value of interaction:", p_value,"\n")

#3d scatterplot
df=groupsDF
df['Location2']=(df['Location']=="Yes").astype(np.float)
threeDcolors = ["darkred","green"]
colors  = [threeDcolors[int(i)] for i in df['Location2']]

fig = plt.figure(0, figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Location2'],df['Coffee'], df['Amplitude'],c=colors, s=60)
ax.plot([0,0],[0,1],[means[3],means[1]],'k-',color="r")
ax.plot([1,1],[0,1],[means[2],means[0]],'k-',color="g")
ax.plot([0,1],[0,0],[means[3],means[2]],'k--')
ax.plot([0,1],[1,1],[means[1],means[0]],'k--')
ax.view_init(15, 220)
ax.set(zlim=(60,110))
ax.set_xticklabels(["","Male","","","","", "Female"])
ax.set_yticklabels(["","","        Yes","","","", "No"])
plt.title("Amplitude 2X2 design:\n Coffee (M/F) , does all exercises (does/doesn't)")
plt.show()


