import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #img = cv2.line(frame, (0,0), (width, height), (255,0,0), 5)
    #parâmetros: imagem fonte, inicio, fim, cor, espessura
    img = cv2.rectangle(frame, (175,175), (400,400), (0, 0, 255), 3)
    #Parâmetros: imagem fonte, posicao da borda superior esquerda, posição da borda inferior direita, cor, finura da linha (-1 para preencher)
    #img = cv2.circle(img, (400,400), 60, (0,255,0), -1)
     
    #drawText
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'Essa garota ama seu namorado', (10, height - 35), font, 1, (0,0,255), 2, cv2.LINE_AA)
    img = cv2.putText(img, 'e esta com saudades dele', (10, height - 10), font, 1, (0,0,255), 2, cv2.LINE_AA)
    
    # Parâmetros: texto, posição central, fonte, escala da fonte, cor, finura da linha, tipo da linha
    
    cv2.imshow('frame', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()