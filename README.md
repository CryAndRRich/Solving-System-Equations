# Solving System Equations
This is a **Flask project** that I got the **idea** from learning about matrices and system of linear equations in university

Basically, when solving systems with a **large number of equations**, the manual methods will be **time-consuming** or **memory-intensive**, this leads to the class of **iterative methods** (Read more [at](https://github.com/CryAndRRich/Solving-System-Equations/blob/main/Matrix/README.md))

Based on that, I created this project with the purpose of plotting **accuracy charts** and comparing the **optimality** between different methods

# Detail
The project uses 3 main **libraries**: `Flask`, `NumPy`, and `Matplotlib`

## SolveEquations.py
This file takes **user-provided data** and **transforms** it into a NumPy array to **represent** a system of equations

Additionally, all NumPy arrays I use are **3-dimensional**, as I convert all numbers into **fractions**. This **ensures** that even if the solution to the system of equations is an **infinitely decimal**, the result remains **accurate**

Then, the function `sovleSLE` is used to **accurately calculate** the solution of the system of equations and return the solution with a **statement** ("No Solution", "Unique Solution" or "Infinite Solution")

## Plot.py
This file takes the **solution** of the system from `SolveEquations.py` and **user-provided data** to choose which **method** needs to be applied (Details about all [methods](https://github.com/CryAndRRich/Solving-System-Equations/blob/main/Matrix/README.md))

The file will then execute, **producing** a plot that **visualizes** the accuracy of the results **compared** to the baseline after each iteration of the method (I use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) for calculating the accuracy)

