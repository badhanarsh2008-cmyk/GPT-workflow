import pandas as pd 

students={
    "Name":[],
    "Roll no.":[],
    "Marks":[]
}

while True:
    students["Name"].append(input("Enter name:"))
    students["Roll no."].append(input("Enter Roll no :"))
    students["Marks"].append(input("Enter Marks:"))
    
    print("\n")
    choice=input("Wanna Exit? (y/n) :" )
    if choice=='y':
        break
    
df=pd.DataFrame(students)
print("Dataframe of Students")
print(df)