from tkinter import *
from tkinter import messagebox
from joblib import load
import numpy as np

regrNO = load("bagging_regressor_NO.joblib")
regrNO2 = load("bagging_regressor_NO2.joblib")
regrO3 = load("bagging_regressor_O3.joblib")
regrPM10 = load("bagging_regressor_PM10.joblib")
scaler = load("scaler.joblib")
root = Tk()
root.title('Predict Value')
root.resizable(0,0)

tempL = Label(root, text = "Temp(C)")
pressL = Label(root, text = "Press(hPa)")
umrL = Label(root, text = "UMR(%)")
noL = Label(root, text = "NO(ug/m3)")
no2L = Label(root, text = "NO2(ug/m3)")
ozoneL = Label(root, text = "Ozone(ug/m3)")
pm10L = Label(root, text = "PM10(ug/m3)")

nonormL = Label(root, text = "NO/NO2 norm is:%d" % 200)
o3normL = Label(root, text = "O3 norm is:%d" % 200)
pm10normL = Label(root, text = "PM10 norm is:%d" % 50)

tempE = Entry(root)
umrE = Entry(root)
pressE = Entry(root)
noE = Entry(root)
no2E = Entry(root)
ozoneE = Entry(root)
pm10E = Entry(root)

tempL.grid(row=0)
tempE.grid(row=0, column=1)

umrL.grid(row=1)
umrE.grid(row=1, column=1)

pressL.grid(row=2)
pressE.grid(row=2, column=1)

noL.grid(row=3)
noE.grid(row=3, column=1)
nonormL.grid(row=3, column=2)

no2L.grid(row=4)
no2E.grid(row=4, column=1)

ozoneL.grid(row=5)
ozoneE.grid(row=5, column=1)
o3normL.grid(row=5, column=2)

pm10L.grid(row=6)
pm10E.grid(row=6, column=1)
pm10normL.grid(row=6, column=2)

def Predict():
    noE.delete(0,END)
    no2E.delete(0,END)
    ozoneE.delete(0,END)
    pm10E.delete(0,END)
    try:
        arr = np.array([[float(tempE.get()), float(umrE.get()), float(pressE.get())]])
        arr_transformed = scaler.transform(arr)
        
        pr_NO = regrNO.predict(arr_transformed)
        pr_NO2 = regrNO2.predict(arr_transformed)
        pr_O3 = regrO3.predict(arr_transformed)
        pr_PM10 = regrPM10.predict(arr_transformed)
        
        noE.insert(0, round(pr_NO[0], 2))
        no2E.insert(0, round(pr_NO2[0], 2))
        ozoneE.insert(0, round(pr_O3[0], 2))
        pm10E.insert(0, round(pr_PM10[0], 2))
    except ValueError:
        messagebox.showinfo("Wrong Value", "Please enter float values!")
        
b = Button(root, text="Predict", command=Predict)

b.grid(row=7)

root.mainloop()
        




    
