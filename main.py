from pytube import Search
from pytubefix import YouTube
import matplotlib.pylab as plt
import librosa
import numpy as np
import os
import csv
from PIL import Image, ImageTk
import splitfolders
import shutil
from ultralytics import YOLO
import wget
import tkinter as tk


def searchVideo(songName):
    """
    Busca un video en YouTube basado en el nombre de una canción.

    Esta función utiliza la clase `Search` para realizar la búsqueda en YouTube.  
    Devuelve un diccionario con información del primer resultado de la búsqueda
    o `None` si no se encuentran resultados.

    Args:
        songName (str): El nombre de la canción a buscar.

    Returns:
        dict: Un diccionario con la información del primer video encontrado.
              El diccionario contiene las siguientes claves:
                - "title" (str): El título del video.
                - "url" (str): La URL del video en YouTube.
                - "image" (str): La URL de la miniatura del video.
              Retorna `None` si no se encuentra ningún video.
    """

    search = Search(songName)
    if len(search.results) == 0: return None
    result = search.results[0]
    return {"title": result.title, "url": result.watch_url, "image": result.thumbnail_url}



def downloadAudio(url, songName):
    """
    Descarga el audio de un video de YouTube.

    Esta función utiliza la librería `pytube` para descargar el audio de un video de YouTube a partir de su URL.  
    El archivo de audio se guarda en formato WAV.

    Args:
        url (str): La URL del video de YouTube.
        songName (str): El nombre del archivo de audio a guardar (sin la extensión .wav).

    Returns:
        None. La función descarga el archivo y no retorna ningún valor.
    """

    yt = YouTube(url, use_oauth=True)
    print(yt.title)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download(filename=f"{songName}.wav")



def getSpectrogram(songName):
    """
    Genera y guarda el espectrograma de una canción en formato PNG.

    Esta función carga un archivo de audio en formato WAV, calcula su espectrograma de Mel,
    lo convierte a decibelios, lo visualiza y guarda como una imagen PNG, 
    y luego elimina el archivo WAV original.

    Args:
        nombre_cancion (str): El nombre de la canción (sin la extensión de archivo).
                           Se asume que el archivo WAV se encuentra en el mismo directorio.

    Returns:
        None. La función guarda la imagen del espectrograma y no retorna ningún valor.
    """

    y, sr = librosa.load(f"{songName}.wav")
 
    print("CANCION CARGADA")
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    print("ESPECTROGRAMA CREADO")
    s_dB = librosa.power_to_db(spectrogram, ref=np.max)
    print("ESPECTROGRAMA CONVERTIDO")

    plt.figure()
    librosa.display.specshow(s_dB)
    plt.savefig(f'{songName}.png', dpi=500)  # Guarda la imagen como PNG, dpi para mejorar la calidad de la imagen
    plt.close()

    os.remove(f"{songName}.wav")



def getSongsList(numberSongsByGenre):
    """
    Carga la lista de canciones desde un archivo CSV o la crea si no existe.

    Esta función intenta leer la lista de canciones desde un archivo llamado "listaCanciones.csv".
    Si el archivo no existe, lo crea a partir de los datos de "spotify_songs.csv", limitando
    el número de canciones por género especificado por `numberSongsByGenre`.

    Args:
        numberSongsByGenre (int): El número máximo de canciones por género a incluir
                                          si se crea el archivo "listaCanciones.csv". 
                                          

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa una canción
              y contiene las claves "title" (título de la canción) y "genre" (género).
    """

    songsList = []

    if os.path.isfile("listaCanciones.csv"):
        fs = open("listaCanciones.csv", mode="r")
        csvFile = csv.reader(fs, delimiter=";")

        for song in csvFile:
            songsList.append({"title": song[0].replace("/", ""), "genre": song[1]})

    else:

        lastGenre = ""
        songsCounter = 0

        fs = open("listaCanciones.csv", "w")

        with open('spotify_songs.csv', mode ='r')as stream:
            csvFile = csv.reader(stream, delimiter=";")

            for song in csvFile:

                if lastGenre != song[1]: 
                    lastGenre = song[1]
                    songsCounter = 1
                    songsList.append({"title": song[0].replace("/", ""), "genre": song[1]})
                    fs.writelines(f"{song[0].replace("/", "")};{song[1]}\n")

                else:
                    if songsCounter < numberSongsByGenre:
                        songsCounter += 1
                        songsList.append({"title": song[0].replace("/", ""), "genre": song[1]})
                        fs.writelines(f"{song[0].replace("/", "")};{song[1]}\n")

    fs.close()
    return songsList



def createDataset():
    """
    Procesa una lista de canciones, descargando audio, generando espectrogramas y organizándolos por género.

    Esta función realiza los siguientes pasos para cada canción en la lista:
        1. Busca el video en YouTube.
        2. Descarga el audio.
        3. Genera el espectrograma.
        4. Mueve el espectrograma a la carpeta correspondiente al género.
        5. Divide el dataset en conjuntos de entrenamiento y validación.

    Args:
        num_canciones (int): El número de canciones a procesar.
    """

    getSongsList(200)
    with open("listaCanciones.csv", encoding="utf-8") as fs:
        listaCanciones = fs.readlines()


    for cancion in listaCanciones:
        titulo, genero = cancion.split(";")
        genero = genero.strip()
        resultado = searchVideo(titulo)
        downloadAudio(resultado["url"], titulo)
        getSpectrogram(titulo)

        if not os.path.isdir(f"espectrogramas/{genero}"):
            os.mkdir(f"espectrogramas/{genero}")
        
        if not os.path.isfile(f"espectrogramas/{genero}/{titulo}.png"): 
            shutil.move(f"{titulo}.png", f"espectrogramas/{genero}")
        else:
            os.remove(f"{titulo}.png")

    splitfolders.ratio("espectrogramas", output="dataset", ratio=(.8, 0.2)) 




def search():
    """
    Realiza una búsqueda de una canción, descarga el audio y el thumbnail, 
    genera el espectrograma, muestra la imagen en la interfaz y realiza una predicción.

    Esta función realiza las siguientes acciones:
        1. Limpia archivos temporales (thumbnail.png, espectrogram.png).
        2. Obtiene el título de la canción desde la entrada `songNameEntry`.
        3. Busca el video en YouTube usando `searchVideo`.
        4. Descarga el audio usando `downloadAudio`.
        5. Genera el espectrograma usando `getSpectrogram`.
        6. Renombra el espectrograma a "espectrogram.png".
        7. Descarga el thumbnail del video o usa una imagen predeterminada si falla la descarga.
        8. Redimensiona y muestra el thumbnail en `thumbnailLabel`.
        9. Redimensiona y muestra el espectrograma en `spectrogramLabel`.
        10. Realiza una predicción usando el modelo `model`.
        11. Muestra los resultados de la predicción en `resultsLabel` y el título de la canción en `completeSongNameLabel`.

    """

    if os.path.exists("thumbnail.png"): os.remove("thumbnail.png")
    if os.path.exists("espectrogram.png"): os.remove("espectrogram.png")
    title = songNameEntry.get()
    result = searchVideo(title)
    downloadAudio(result["url"], title)
    getSpectrogram(title)
    os.rename(f"{title}.png", "espectrogram.png")


    try :
        wget.download(result["image"], out="thumbnail.png")
        image = Image.open("thumbnail.png")
        image = image.resize((image.size[0]//5, image.size[1]//5))
    except :
        image = Image.open("noThumbnail.png")
        image = image.resize((image.size[0]//5, image.size[1]//5))

    thumbnail = ImageTk.PhotoImage(image)
    thumbnailLabel.config(image=thumbnail)
    thumbnailLabel.image = thumbnail
    

    image = Image.open("espectrogram.png")
    image = image.resize((image.size[0]//15, image.size[1]//15))
    spectrogram = ImageTk.PhotoImage(image)
    spectrogramLabel.config(image=spectrogram)
    spectrogramLabel.image = spectrogram


    results = model("espectrogram.png")[0]

    stringResult = ""

    for index, element in enumerate(results.probs.top5):
        stringResult += model.names[element] + " "
        stringResult += str(round(results.probs.top5conf.tolist()[index]*100,2)) + "%    "

    completeSongNameLabel.config(text=result["title"])
    resultsLabel.config(text=stringResult)



# Entrenamiento del modelo
# model = YOLO("YOLO11x-cls.pt")
# results = model.train(data="dataset", epochs=30, patience=5, device="mps")


if __name__ == "__main__":
    """
    Crea la interfaz gráfica para la clasificación de canciones.

    Esta sección del código se ejecuta solo cuando el script se ejecuta directamente.
    Crea la ventana principal de la aplicación,
    los widgets (entrada de texto, botones, labels) y configura el layout.  También
    carga el modelo YOLO para la clasificación.

    """

    model = YOLO("clasifyModel.pt")
    model("noThumbnail.png")

    root = tk.Tk()
    root.title("SONGS CLASIFY")
    root.geometry("600x400")

    searchFrame = tk.Frame(root)
    searchFrame.pack(pady=20)
    songNameEntry = tk.Entry(searchFrame)
    songNameEntry.pack(side=tk.LEFT)
    searchButton = tk.Button(searchFrame, text="Buscar", command=search)
    searchButton.pack(side=tk.LEFT)


    imageFrame = tk.Frame(root)
    imageFrame.pack(pady=20)
    thumbnailLabel = tk.Label(imageFrame)
    spectrogramLabel = tk.Label(imageFrame)
    thumbnailLabel.pack(side="left", padx=10)
    spectrogramLabel.pack(side="left", padx=10)


    completeSongNameLabel = tk.Label(root)
    completeSongNameLabel.pack(pady=10)


    resultsLabel = tk.Label(root)
    resultsLabel.pack(pady=10)

    root.mainloop()