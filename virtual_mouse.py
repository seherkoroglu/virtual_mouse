import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

# Ekran boyutları
ekran_genislik = 1280
ekran_yukseklik = 720

# Klavye tuşları
klavye_tuslar = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", "GERİ"],
    ["BOŞLUK", "KAYDET"]
]

# Başlangıç metni
girdi_metin = ""

# Buton boyutları
buton_genislik = 80
buton_yukseklik = 80
opaklik = 0.7

# Mediapipe el takip ayarları
mp_eller = mp.solutions.hands
eller = mp_eller.Hands(max_num_hands=1)
mp_ciz = mp.solutions.drawing_utils

# Pygame ile ses başlat
pygame.mixer.init()

def play_click_sound():
    pygame.mixer.music.load("click_sound.mp3")
    pygame.mixer.music.play()

# Video akışını başlat
cap = cv2.VideoCapture(0)
cap.set(3, ekran_genislik)
cap.set(4, ekran_yukseklik)

# Tıklama zamanı ve aktif tuşu takip için değişkenler
son_gezinilen_tus = None
gezinme_baslangic_zamani = None
gezinme_suresi_esigi = 2  # Saniye cinsinden tıklama süresi


def klavyeyi_ciz(resim, tuslar, girdi_metin, gezinilen_tus=None):
    global buton_genislik, buton_yukseklik, opaklik
    x, y = 50, 100

    # Opaklık efekti için maske oluştur
    kaplama = resim.copy()

    # Metin kutusunu çiz
    cv2.rectangle(resim, (50, 30), (ekran_genislik - 50, 90), (50, 50, 50), -1)
    cv2.putText(resim, girdi_metin, (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for satir in tuslar:
        for tus in satir:
            # Butonu çiz
            if tus == "BOŞLUK":
                w = buton_genislik * 5  # BOŞLUK tuşu için genişlik
            else:
                w = buton_genislik

            # Pembe hover efekti
            if tus == gezinilen_tus:
                cv2.rectangle(kaplama, (x, y), (x + w, y + buton_yukseklik), (255, 0, 255), -1)  # Pembe renk
            else:
                cv2.rectangle(kaplama, (x, y), (x + w, y + buton_yukseklik), (200, 200, 200), -1)  # Normal renk

            cv2.putText(kaplama, tus, (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            x += w + 10  # Tuşlar arasına boşluk ekle

        x = 50  # Sıradaki satır için x'i sıfırla
        y += buton_yukseklik + 10

    # Opaklık efekti uygulama
    cv2.addWeighted(kaplama, opaklik, resim, 1 - opaklik, 0, resim)


# Parmağın belirli bir buton üzerinde olup olmadığını kontrol etme
def buton_gezinme_kontrol(x, y, tuslar):
    btn_x, btn_y = 50, 100
    for satir in tuslar:
        for tus in satir:
            # Buton genişliğini ayarla
            w = buton_genislik * 5 if tus == "BOŞLUK" else buton_genislik

            # Parmağın bu butonun üstünde olup olmadığını kontrol et
            if btn_x <= x <= btn_x + w and btn_y <= y <= btn_y + buton_yukseklik:
                return tus
            btn_x += w + 10

        btn_x = 50  # Sıradaki satır için x'i sıfırla
        btn_y += buton_yukseklik + 10

    return None


def dosyaya_kaydet(metin):
    # Kaydedilecek dosya adı
    dosya_adı = "metin_dosyasi.txt"
    with open(dosya_adı, "w") as dosya:
        dosya.write(metin)
    print(f"{dosya_adı} dosyasına kaydedildi.")


while True:
    ret, kare = cap.read()
    kare = cv2.flip(kare, 1)
    resim_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    # El takibi
    sonuc = eller.process(resim_rgb)
    gezinilen_tus = None
    if sonuc.multi_hand_landmarks:
        for el_isaretleri in sonuc.multi_hand_landmarks:
            # İşaret parmağı ucunun koordinatlarını al
            parmak_ucu_x = int(el_isaretleri.landmark[mp_eller.HandLandmark.INDEX_FINGER_TIP].x * ekran_genislik)
            parmak_ucu_y = int(el_isaretleri.landmark[mp_eller.HandLandmark.INDEX_FINGER_TIP].y * ekran_yukseklik)

            # Parmağın üstünde olduğu butonu kontrol et
            gezinilen_tus = buton_gezinme_kontrol(parmak_ucu_x, parmak_ucu_y, klavye_tuslar)

            # Tıklama algılama
            if gezinilen_tus:
                if gezinilen_tus == son_gezinilen_tus:
                    # Aynı tuş üzerinde 2 saniyedir duruyorsa
                    if gezinme_baslangic_zamani and (time.time() - gezinme_baslangic_zamani >= gezinme_suresi_esigi):
                        play_click_sound()  # Play sound when a button is clicked
                        if gezinilen_tus == "GERİ":
                            girdi_metin = girdi_metin[:-1]
                        elif gezinilen_tus == "BOŞLUK":
                            girdi_metin += " "
                        elif gezinilen_tus == "KAYDET":
                            dosyaya_kaydet(girdi_metin)  # Kaydet butonuna tıklandığında metni kaydet
                        else:
                            girdi_metin += gezinilen_tus
                        gezinme_baslangic_zamani = None  # Tıklamayı sıfırla
                else:
                    son_gezinilen_tus = gezinilen_tus
                    gezinme_baslangic_zamani = time.time()  # Yeni tuşa geçildiği için zamanı sıfırla
            else:
                son_gezinilen_tus = None
                gezinme_baslangic_zamani = None

            # İşaret parmağı ucunu çiz
            cv2.circle(kare, (parmak_ucu_x, parmak_ucu_y), 10, (255, 0, 0), -1)

            # Landmark çizimi
            mp_ciz.draw_landmarks(kare, el_isaretleri, mp_eller.HAND_CONNECTIONS)

    # Klavye ve metin kutusunu çiz
    klavyeyi_ciz(kare, klavye_tuslar, girdi_metin, gezinilen_tus)

    # Son görüntüyü göster
    cv2.imshow("Sanal Klavye", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
