import pandas as pd

# تحميل البيانات
data = pd.read_csv("data/plant_health_data.csv")

# عرض أسماء الأعمدة
print("🧾 أسماء الأعمدة:")
print(data.columns)

print("\n")

# عرض أول 5 صفوف
print("📊 أول 5 صفوف من البيانات:")
print(data.head())
print("\n")

# عرض معلومات تفصيلية عن كل عمود
print("ℹ️ معلومات عن نوع البيانات في كل عمود:")
print(data.info())

# عرض وصف إحصائي للأعمدة الرقمية
print("\n📈 ملخص إحصائي للأعمدة الرقمية:")
print(data.describe())
