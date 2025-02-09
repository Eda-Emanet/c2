FROM python:3.10-slim

WORKDIR /APP

# Gerekli paketleri yükleyin
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Tüm proje dosyalarını kopyalayın (notebook dosyanız dahil)
COPY . .

# Jupyter Notebook sunucusunu kurun (requirements.txt içinde yoksa ekleyin)
RUN python -m pip install jupyter

# Container portunu açın
EXPOSE 8888

# Jupyter Notebook'u başlatın (localhost'a erişimi sağlamak için --ip=0.0.0.0 kullanıyoruz)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]