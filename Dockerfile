# Gebruik officiële Python 3.11 image als basis
FROM python:3.11-slim

# Werkdirectory in de container
WORKDIR /app

# Kopieer requirements.txt en installeer dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Kopieer alle bestanden naar de werkdirectory
COPY . .

# Expose de poort waarop FastAPI draait
EXPOSE 5000

# Start de app met uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
