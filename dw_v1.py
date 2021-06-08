import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import svm, preprocessing
from sklearn.cluster import KMeans
from statsmodels.stats.multicomp import pairwise_tukeyhsd


######### CONFIG #########

sns.set_theme(style = "darkgrid")
sns.color_palette("flare", as_cmap = True)


######### BASICS #########

def excel_check(file):
	if ".xlsx" in file:
		print("Detected an Excel file, boss. \n")
		worksheet = int(input("Which worksheet number to load, Boss? \n"))
		df = pd.read_excel(file, sheet_name = worksheet)
		return df
	else:
		df = pd.read_csv(file, decimal = ".", delimiter = ",")
		return df

def info():
	print("\nHere are the basics about this data, Boss: \n")
	print(df.info())

def head():
	print(df.head(10))

def tail():
	print(df.tail(10))

def desc():
	print(df.describe())

def count():
	print(df.value_counts())

def count_missing():
	print("\nI've counted these missing values, Boss: \n")
	print(df.isna().sum())


######### DATAFRAME JUGGLEZ #########

def multi_select():
	payload = []
	state = True

	while (state):
		entry = input("Which column to select, boss? (Press XX for pushing the payload.) \n")

		if (entry == "XX"):
			state = False
			print("\nAllright, Boss – I'm pushing the payload to process!\n")	
		else:
			payload.append(entry)
			print("Added this column to the payload, Boss!\n")
			print("Current payload:")
			print(payload)
			print("\n")

	return payload

def load_new_df():
	payload = multi_select()
	return df_save[payload]

def remove_column():
	payload = multi_select()
	return df.drop(payload, axis = 1)

def pivot(column, row, value):
	return df.pivot(columns = column, index = row, values = value)

def crosstab(x, y):
	return pd.crosstab(df[x], df[y])

def drop_dup():
	return df.drop_duplicates().reset_index(drop = True)

def drop_nan():
	return df.dropna().reset_index(drop = True)

def fill_nan(value):
	return df.fillna(value)

def normalize():
	x = df.values
	min_max_scaler = preprocessing.MinMaxScaler()
	x_normalized = min_max_scaler.fit_transform(x)
	return pd.DataFrame(x_normalized, columns = df.columns)


######### VISUALS #########	

def plot(y):
	sns.lineplot(x = range(len(df)), y = y, data = df)

	mean = df[y].mean()
	std = df[y].std()
	std_lo = mean - std
	std_hi = mean + std

	plt.axhline(mean, 0, 1, color = "red", label = "Mean")
	plt.axhline(std_lo, 0, 1, color = "black", linestyle = "--")
	plt.axhline(std_hi, 0, 1, color = "black", linestyle = "--")

	plt.title(y)
	plt.legend()

	plt.show()

def scat(x, y):

	sns.regplot(x = x, y = y, data = df)

	mean = df[y].mean()
	std = df[y].std()
	std_lo = mean - std
	std_hi = mean + std

	plt.axhline(mean, 0, 1, color = "red", label = "Mean")
	plt.axhline(std_lo, 0, 1, color = "black", linestyle = "--")
	plt.axhline(std_hi, 0, 1, color = "black", linestyle = "--")

	plt.title(x + " vs " + y)
	plt.show()

def heat(x, y, z):

	df_pivot = df.pivot(x, y, z)

	f, ax = plt.subplots(figsize=(9, 6))

	sns.heatmap(df_pivot, annot=True, linewidths=.5, ax=ax)

	plt.title("Heatmap of " + x + "," + y + "," + z)
	plt.show()

def box(y):
	if (y == "All" or y == "all"):
		sns.boxplot(data = df)
	else:
		sns.boxplot(data = df, y = y)

	plt.show()

def violin(y):
	if (y == "All" or y == "all"):
		sns.violinplot(data = df)
	else:
		sns.violinplot(data = df, y = y)

	plt.show()

def hist(y):

	if (y == "All" or y == "all"):
		sns.histplot(data = df, kde = True)
	else:
		sns.histplot(data = df, x = y, kde = True)

		mean = df[y].mean()
		std = df[y].std()
		std_lo = mean - std
		std_hi = mean + std

		plt.axvline(mean, 0, 1, color = "red", label = "Mean")
		plt.axvline(std_lo, 0, 1, color = "black", linestyle = "--")
		plt.axvline(std_hi, 0, 1, color = "black", linestyle = "--")

	plt.show()

def pairplot():
	sns.pairplot(df)

	plt.show()


######### TESTS #########

def side_switch():
	switch = input("""What kind of test are we running, Mr.Boss? \n
		<TwoSided>      Is the observed X different than our expectation?
		<LeftSided>     Is the observed X smaller than our expectation?
		<RightSided>    Is the observed X bigger than our expectation? \n
		""")

	if (switch == "LeftSided"):
		return "LeftSided"
	elif (switch == "RightSided"):
		return "RightSided"
	else:
		return "TwoSided" 

def pval_check_ttest(tstat, pval, sig, test_side):

		if (test_side == "LeftSided"):
			if ((pval / 2) < sig and tstat < 0):
				print("P-Value is: " + str((pval / 2)) + ". T-Value is: " + str(tstat) + ". Result is significant, Boss!")
				print("This means, that the observed X is smaller than our expectation, boss!")
			else:
				print("P-Value is: " + str((pval / 2)) + ". T-Value is: " + str(tstat) + ". Result is insignificant, Boss!")
				print("This means, that the observed X is not smaller than our expectation, boss!")

		elif (test_side == "RightSided"):
			if ((pval / 2) < sig and tstat > 0):
				print("P-Value is: " + str((pval / 2)) + ". T-Value is: " + str(tstat) + ". Result is significant, Boss!")
				print("This means, that the observed X is bigger than our expectation, boss!")
			else:
				print("P-Value is: " + str((pval / 2)) + ". T-Value is: " + str(tstat) + ". Result is insignificant, Boss!")
				print("This means, that the observed X is not bigger than our expectation, boss!")

		else:

			if (pval < sig):
				print("P-Value is: " + str(pval) + ". Result is significant, Boss!")
				print("This means, that the observed X is different than our expectation, boss!")

			else:
				print("P-Value is: " + str(pval) + ". Result is insignificant, Boss!")
				print("This means, that the observed X is not different than our expectation, boss!")

def pval_check(pval, sig):
	if (pval < sig):
		print("P-Value is: " + str(pval) + ". Result is significant, Boss!")
		print("This means, that the observed values are different, boss! \n")
	else:
		print("P-Value is: " + str(pval) + ". Result is insignificant, Boss!")
		print("This means, that the observed values are not different, boss! \n")

def corr(x, y):
	corr, p = stats.pearsonr(df[x], df[y])
	print(corr)

def ttest_1samp(sample_distribution, expected_mean, sig, test_side):
	tstat, pval = stats.ttest_1samp(df[sample_distribution], expected_mean)
	pval_check_ttest(tstat, pval, sig, test_side)

def ttest_2samp(sample_a, sample_b, sig, test_side):
	tstat, pval = stats.ttest_ind(df[sample_a], df[sample_b])
	pval_check_ttest(tstat, pval, sig, test_side)

def anova(sample_a, sample_b, sample_c, sig):
	fstat, pval = stats.f_oneway(df[sample_a], df[sample_b], df[sample_c])
	pval_check(pval, sig)

def tukey(sample_a, sample_b, sig):
	tukey_results = pairwise_tukeyhsd(df[sample_a], df[sample_a], sig)
	print(tukey_results)

def chi(sample_a, sample_b, sig):
	xtable = pd.crosstab(df[sample_a], df[sample_b])
	chi2, pval, dof, expected = stats.chi2_contingency(xtable)
	pval_check(pval, sig)

def bino(successes, expected_probability, sig):
	suc_res = np.sum(df[successes] == 1)
	n = len(df[successes])
	pval = stats.binom_test(suc_res, n, p = expected_probability)
	pval_check(pval, sig)


######### MACHINE INTELLIGENCE #########

def linreg(x, y, predvalue):
	line_fitter = LinearRegression()

	x_in = df[x].values.reshape(-1, 1)
	y_in = df[y].values.reshape(-1, 1)

	line_fitter.fit(x_in, y_in)

	a = round(line_fitter.intercept_[0], 2)
	b = round(line_fitter.coef_[0][0], 2)

	print("\nLinear Regression formula for this model is:")
	print("Y = " + str(a) + " + " + str(b) + "x \n")

	pred_in = np.array(predvalue).reshape(-1, 1)
	y_predicted = round(line_fitter.predict(pred_in)[0][0], 2)

	print("With " + str(predvalue) + " (" + x + ") we expect " + str(y_predicted) + " (" + y + "), Boss!")

def svm_run(sample_a, predvalue):
	y = df[sample_a]
	X = df.drop(columns = sample_a, axis = 1)

	SVM = svm.LinearSVC()
	SVM.fit(X, y)
	print("\nSVM is fit, Boss!\n")

	print("Mean accuracy of the training data is:")
	
	score = SVM.score(X,y)
	print(round(score, 4))

	pred = SVM.predict(predvalue)	
	print("\nPredicted label is:")
	print(str(int(pred[0])) + "\n")

def cluster(clusters):

	X = df.values

	kmeans = KMeans(n_clusters = clusters)
	kmeans.fit(X)
	print(kmeans.cluster_centers_)

	to_plot = input("\nBoss, do you want see the plot for these clusters? (Yes / No) \n")

	if (to_plot == "Yes"):

		plt.scatter(X[:,0],X[:,1], c = kmeans.labels_)

		columns = df.columns

		plt.xlabel(columns[0])
		plt.ylabel(columns[1])
		plt.title(columns[0] + " vs " + columns[1] + " Cluster")

		plt.show()


######### BONUS #########

def calc_samplesize(std, aim, baseline):
	z = 1.96
	sig = 0.05
	power = 0.8416
	std = std
	mde = (aim - baseline) / 100
	print("\nMinimum detectable effect is " + str(mde * 100) + " %.")

	n = ( 2 * ((z + power) * 2) * ((std) * 2) ) / ( (mde) * 2 )
	n = round(n)
 
	print("Sample size of " + str(n) + " is required, Boss!")


######### ENGINE #########

def help():

	print("""
*********************************************************************************************************************************	
* * * * * *                                               ◊ DATAWIZ ◊                                                 * * * * * *
* *                                           v1.0             ◊                                                              * *
*                                                                                                                               *
*                                  by David Weicht – Project Manager for Digital Marketing                                      *
*                                                     www.davidweicht.de                                                        *
*********************************************************************************************************************************
*                                                                                                                               *
*    COMMANDS   ______________________________________________________________________________________________________________  * 
*                                                                                                                               *
*                                                                                                                               *
*    _BASICS                                  _PLOTS                                 _JUGGLEZ                                   *
*                                                                                                                               *
*    <Info>     Data Meta                     <Plot>     Lineplot                    <New>      New DataFrame                   *
*    <Head>     Data Heads                    <Scat>     Scatterplot                 <Reload>   Reload Original DataFrame       *
*    <Tail>     Data Tails                    <Heat>     Heatmap                     <Delete>   Delete Column(s)                *
*    <Desc>     Descriptive Stats             <Box>      Boxplot                     <Pivot>    Pivot DataFrame                 *
*    <Count>    Count Values                  <Violin>   Violinplot                  <Cross>    Create Crosstable               *
*    <CountM>   Count Missing Values          <Hist>     Histogram                   <DropD>    Drop Duplicates                 *
*                                             <Pair>     Pairplot                    <DropN>    Drop NaN                        *
*                                                                                    <FillN>    Fill NaN                        *
*                                                                                    <Norm>     Normalize DataFrame             *
*                                                                                                                               *
*                                                                                                                               *
*    _ANALYSIS                                _MACHINE INTELLIGENCE                  _BONUS                                     *
*                                                                                                                               *
*    <Corr>     Correlation                   <LinR>     Linear Regression           <Size>     Sample Size Calculator          *
*    <Test1>    T-Test (One Sample)           <SVM>      SVM (Two Samples)                                                      *
*    <Test2>    T-Test (Two Samples)          <Cluster>  K-Means Clustering                                                     *
*    <Anova>    Anova (Three Samples)                                                                                           *
*    <Tukey>    Tukey-Test                                                                                                      *
*    <Chi2>     Chi-Square-Test                                                                                                 *
*    <Bino>     Binomial Test                                                                                                   *
*                                                                                                                               *
*                                                                                                                               *
*    <Help>     Commands                                                                                                        *
*    <Quit>     Quit Datawiz                                                                                                    *
*                                                                                                                               *
*********************************************************************************************************************************
	""")

def commander(cmd):
	global df, df_save

	if (cmd == "Info" or cmd == "info"):
		info()

	elif (cmd == "Head" or cmd == "head"):
		head()

	elif (cmd == "Tail" or cmd == "tail"):
		tail()

	elif (cmd == "Desc" or cmd == "desc"):
		desc()

	elif (cmd == "Count" or cmd == "count"):
		count()

	elif (cmd == "CountM" or cmd == "countm"):
		count_missing()

	elif (cmd == "Plot" or cmd == "plot"):
		y = input("Which column to plot as Y, Boss? \n")
		plot(y)

	elif (cmd == "Scat" or cmd == "scat"):
		x = input("Which column to plot as X, Boss? \n")
		y = input("Which column to plot as Y, Boss? \n")
		scat(x, y)

	elif (cmd == "Heat" or cmd == "heat"):
		x = input("Which column to plot as X, Boss? \n")
		y = input("Which column to plot as Y, Boss? \n")
		z = input("Which column to plot as Z, Boss? \n")
		heat(x, y, z)

	elif (cmd == "Box" or cmd == "box"):
		y = input("Which column(s) to plot, Boss? (All for all columns.) \n")
		box(y)

	elif (cmd == "Violin" or cmd == "violin"):
		y = input("Which column(s) to plot, Boss? (All for all columns.) \n")
		violin(y)

	elif (cmd == "Hist" or cmd == "hist"):
		y = input("Which column(s) to plot, Boss? (All for all columns.) \n")
		hist(y)

	elif (cmd == "Pair" or cmd == "pair"):
		pairplot()

	elif (cmd == "Corr" or cmd == "corr"):
		x = input("Which column to plot as X, Boss? \n")
		y = input("Which column to plot as Y, Boss? \n")
		corr(x, y)

	elif (cmd == "Test1" or cmd == "test1"):
		sample_distribution = input("What is the sample, Boss? \n")
		expected_mean = float(input("What is the expected mean, Boss? \n"))
		test_side = side_switch()
		sig = float(input("What significance level, Boss? \n"))
		ttest_1samp(sample_distribution, expected_mean, sig, test_side)

	elif (cmd == "Test2" or cmd == "test2"):
		sample_a = input("What is first sample group, Boss? \n")
		sample_b = input("What is second sample group, Boss? \n")
		test_side = side_switch()
		sig = float(input("What significance level, Boss? \n"))
		ttest_2samp(sample_a, sample_b, sig, test_side)

	elif (cmd == "Anova" or cmd == "anova"):
		sample_a = input("What's the first sample group, Boss? \n")
		sample_b = input("What's the second sample group, Boss? \n")
		sample_c = input("What's the third sample group, Boss? \n")		
		sig = float(input("What's the significance level, Boss? \n"))
		anova(sample_a, sample_b, sample_c, sig)

	elif (cmd == "Tukey" or cmd == "tukey"):
		sample_a = input("What's the float sample, Boss? \n")
		sample_b = input("What's the grouped sample, Boss? \n")	
		sig = float(input("What's the significance level, Boss? \n"))
		tukey(sample_a, sample_b, sig)

	elif (cmd == "Chi2" or cmd == "chi2"):
		sample_a = input("What is first sample group, Boss? \n")
		sample_b = input("What is second sample group, Boss? \n")
		sig = float(input("What significance level, Boss? \n"))
		chi(sample_a, sample_b, sig)

	elif (cmd == "Bino" or cmd == "bino"):
		successes = input("Which column shall I take for successes, Boss? \n")
		expected_probability = float(input("What is expected probability, Boss? \n"))
		sig = float(input("What significance level, Boss? \n"))
		bino(successes, expected_probability, sig)

	elif (cmd == "LinR" or cmd == "linr"):
		x = input("Which column for X, Boss? \n")
		y = input("Which column for Y, Boss? \n")
		predvalue = float(input("Which x-value to predict a y-value, Boss? \n"))
		linreg(x, y, predvalue)

	elif (cmd == "SVM" or cmd == "svm"):

		print("\nBoss, you remember to better normalize that data first? \n")

		y = input("Which column is featuring the label to be classified, Boss? \n")
		pred_a = input("What's the first value to predict, boss? \n")
		pred_b = input("What's the second value to predict, boss? \n")
		predvalue = np.array([[float(pred_a), float(pred_b)]])
		svm_run(y, predvalue)

	elif (cmd == "Cluster" or cmd == "cluster"):
		clusters = int(input("How many clusters to try, Boss? \n"))
		cluster(clusters)

	elif (cmd == "Size" or cmd == "size"):
		aim = float(input("What aim (in XX %) do we want to reach, Boss? \n"))
		baseline = float(input("What's the baseline (in XX %), Boss? \n"))
		std = float(input("What's the standard deviation (in XX %), Boss? \n"))		
		calc_samplesize(std, aim, baseline)

	elif (cmd == "New" or cmd == "new"):
		df = load_new_df()
		print("New DataFrame created, boss!")

	elif (cmd == "Reload" or cmd == "reload"):
		df = df_save
		print("Original DataFrame reloaded, boss!")

	elif (cmd == "Delete" or cmd == "delete"):
		df = remove_column()
		print("Deleted, Boss!\n")

	elif (cmd == "Pivot" or cmd == "pivot"):
		column = input("Which column to pivot, Boss? \n")
		row = input("Which column to take as rows, boss? \n")
		value = input("Which column to take as values, Boss? \n")	
		df = pivot(column, row, value)
		print("Pivot done, boss!\n")

	elif (cmd == "Cross" or cmd == "cross"):
		x = input("Which column for X, Boss? \n")
		y = input("Which column for Y, Boss? \n")
		df = crosstab(x, y)
		print(df)

	elif (cmd == "DropD" or cmd == "dropd"):
		df = drop_dup()
		print("Dropped the dups, Boss!\n")

	elif (cmd == "DropN" or cmd == "dropn"):
		df = drop_nan()
		print("Dropped the NaNs, Boss!\n")

	elif (cmd == "FillN" or cmd == "filln"):
		fill_value = input("\nWhat to fill the missing values with, Boss? \n")
		df = fill_nan(fill_value)
		print("\nFilled all NaNs with " + str(fill_value) + ", boss!\n")

	elif (cmd == "Norm" or cmd == "norm"):
		df = normalize()
		print("Normlized the data, boss!\n")

	elif (cmd == "Help" or cmd == "help"):
		help()

	elif (cmd == "Quit" or cmd == "quit"):
		print("Allright – I'm out, Boss! \n")
		exit()

	else:
		print("I'm sorry, Boss. I'm stupid. Dunno whattu do...")

def run():
	help()

	while True:
		cmd = input("What now, Boss? \n")
		commander(cmd)
		print("\n")


######### RUNNER #########

file = "./data.csv"

df = excel_check(file)
df_save = df

run()



######### IDEEN #########
#
# Check des Sample Size Rechners
#
# Weiteres Machine Learning
#	SVM (Multivariater Input).
#	Multi-Class Classifier.
#   PyTorch.
#
# Detailed Plot Selector
#	Select Plot-Type
#		Weitere Plots
#			Swarm
#			Countplot
#	Select X-Value
#   Select Y-Value
#	Select Variable as Hue
#
# Aktualisiere Tests mit Alternativen-Testseitenparameter
#
# Anova mit Multiauswahl von Variablen
#
# Data Science Wiki
#
# Input-Fehler ableiten
#
# Panda DataFrame Operationen
#	Füge kalkulierte Spalte hinzu
#		Inklusive Text-Input als Operation (eval("2 + 2"))
#
# GeoMaps
#
# CI-Extractor (zieht z.B. Farbcodes einer URL und erstellt ein Seaborn-Farbschema)
#
# ColorScheme Selector
#
# SQL-Uploader
#
# Tableau-API-Connector
#
# Webscraper
#
# Allround Report als PDF
#	Save to Report
#	Print Report as PDF
# 
# Mailer
#
# Ablage weiterer Dataframes
#	Automatische Aufteilung in Trainings- und Test-Daten.
# Übersicht aktueller Dataframes
# Auswahl bestimmter Dataframes
# 
# Plot auf der Website anzeigen
#
# 
