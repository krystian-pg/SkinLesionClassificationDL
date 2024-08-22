FROM pytorch/pytorch:latest

# Creating main directory
WORKDIR /app

# Copying packages to install 
COPY requirements.txt ./

# Installing all necessary packages
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copying current directory to main container direcotyr
COPY . .