# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:34:53 2018

@author: Liz
"""
#--AGREGAR AL CÓDIGO ORIGINAL--
from skimage.feature import peak_local_max,canny
from skimage.feature import match_template
from skimage.io import imread
from skimage.color import rgb2gray

#from skimage.transform import resize
from skimage.filters.thresholding import threshold_otsu
from skimage.exposure import rescale_intensity
 
 # In[]
plt.close('all')
gen = 'Imagenes/'
folder_ids = os.listdir(gen)
#ID's de las imágenes
image_ids = list()

#t_name = 'template1'#Nombre del template a usar
r = 50 #radio de referencia
r_pozo = 110 #radio del pozo
template = circle(r)

#Creacíon de una matriz que contenga todos los ids, donde por cada fila hay 1 carpeta.
for i in range(0,len(folder_ids)):
    image_ids.insert(i, os.listdir(gen + folder_ids[i] + '/') )    
#for i in range(0,len(folder_ids)):
i = 2#número del folder
for j in range(0,len(image_ids[i][:])):
    PATH = gen + folder_ids[i] + '/' + image_ids[i][j]
    I = imread(PATH)
    I_gray = rgb2gray(I)
    I_edges = canny(I_gray,sigma=0.05)
    coord = corr2d(I_edges,template,folder_ids[i],image_ids[i][j],j,r_pozo)
    I_gray = rescale_intensity(I_gray,in_range=(0.1,0.8))
    m = np.shape(I_gray)
    I_final = np.ones([m[0],m[1]])
    for k in range(0,len(coord)):
#        k = 0
       I_crop = pozos(I_gray,coord[k],r_pozo)
       I_otsu = otsu(I_crop,coord[k],r_pozo)
       I_final = I_final * I_otsu
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(I_final,cmap='gray')
    fig.savefig('Resultados_otsu/' + folder_ids[i] + '/' + image_ids[i][j])
#for i in [3,5]:
##i = 1#número del folder
#    for j in range(0,len(image_ids[i][:])):
#        PATH = gen + folder_ids[i] + '/' + image_ids[i][j]
#        #    offset = 150
#        I = imread(PATH)
#        I_gray = rgb2gray(I)
#        coord = corr2d(I_gray,t_name,folder_ids[i],image_ids[i][j],j)

#template = imread(t_name + '.png')
#template_g = rgb2gray(template) 
 
#PATH = gen + folder_ids[0] + '/' + image_ids[0][7]
#I = imread(PATH)
#I_gray = rgb2gray(I)
#print(np.mean(I_gray[:]))
#I_gray = I_gray - np.mean(I_gray)
# In[]
PATH = gen + folder_ids[1] + '/' + image_ids[1][0]
I = imread(PATH)
I_gray = rgb2gray(I)    
I_edges = canny(I_gray,sigma=0.05)
coord = corr2d(I_edges,template,folder_ids[1],image_ids[1][0],0,r_pozo)
I_gray = rescale_intensity(I_gray,in_range=(0.2,0.8))
m = np.shape(I_gray)
I_final = np.ones([m[0],m[1]])
for k in range(0,len(coord)):
    I_crop = pozos(I_gray,coord[k],r_pozo)
    I_otsu = otsu(I_crop,coord[k],r_pozo)
    I_final = I_final * I_otsu
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.axis('off')
ax1.imshow(I_final,cmap='gray')
# In[]   
def otsu(I_gray,centro,r):
    I = I_gray[(centro[1]-r):(centro[1]+r),(centro[0]-r):(centro[0]+r)]
    prom = np.mean(I)
    thresh = threshold_otsu(I)*prom
    BW = I_gray > thresh
    BW = 1 - BW
    return BW 
   
 
# In[] Correlación cruzada normalizada de skimage
#template = imread('template_re2.png')
#template_g = rgb2gray(template)
#template_g = template_g -  np.mean(template_g)
#result = match_template(I_gray,template_g,pad_input = True)
###máximos locales de la correlación cruzada para encontrar otras coincidencias
#coordinates = peak_local_max(result, min_distance=125)
##
#fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
#ax_orig.imshow(I_gray, cmap='gray')
#ax_orig.set_title('Original')
#ax_orig.set_axis_off()
#ax_template.imshow(template_g, cmap='gray')
#ax_template.set_title('Template')
#ax_template.set_axis_off()
#ax_corr.imshow(result, cmap='gray')
#ax_corr.set_title('Correlación cruzada')
#ax_corr.set_axis_off()
#for i in range(0,len(coordinates)):
#    ax_orig.plot(coordinates[i,1], coordinates[i,0], 'ro')
#fig.show()
 
 
#--Entrada: Imagen en escala de grises I_gray y template para hacer la
#@@ -86,46 +93,46 @@
 #Guarda las imágenes qué dieron como resultado del template empleado en la
 # la dirección Resultados_template/nombre_folder/nombre_imagen
     
#def corr2d(I_gray,t_name,folder,name,ind):
#    template = imread(t_name + '.png')
#    template_g = rgb2gray(template)
def corr2d(BW,template,folder,name,ind,r_pozo):
    result = match_template(I_edges,template,pad_input = True)
    coordinates = peak_local_max(result, min_distance=160)#centro del pequeño 
    if coordinates[0][1] < coordinates[1][1]:
        y = coordinates[0][0]-r_pozo
        x = coordinates[0][1]+r_pozo
    else:
        y = coordinates[1][0]-r_pozo
        x = coordinates[1][1]+r_pozo
    #coordenadas de los otros pozos por simetría
    centros = [[x,y],[x,y+2*r_pozo]]
    for i in range(0,2):
        centros.append([centros[i][0] + 2*r_pozo,centros[i][1] ])
        centros.append([centros[i][0] - 2*r_pozo,centros[i][1] ])

    centros = sorted(centros)
#    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
#    fig = plt.figure()
#    ax_orig = fig.add_subplot(111)
##    ax_orig.imshow(I_gray, cmap='gray')
#    ax_orig.imshow(BW, cmap='gray')
#    ax_orig.set_title('Original')
#    ax_orig.set_axis_off()
##    ax_template.imshow(template, cmap='gray')
##    ax_template.set_title('Template')
##    ax_template.set_axis_off()
##    ax_corr.imshow(result, cmap='gray')
##    ax_corr.set_title('Correlación cruzada')
##    ax_corr.set_axis_off()
##    for i in range(0,len(coordinates)):
##        ax_orig.plot(coordinates[i,1], coordinates[i,0], 'ro')
#    for i in range(0,len(centros)):
#        ax_orig.plot(centros[i][0],centros[i][1],'ro')
#    fig.show()   
#    fig.savefig('Resultados_template/' + folder + '/' + name)
#    fig.savefig('Resultados_template/' + t_name + '/' + folder + '/' + name)
    return centros
    
#coord = corr2d(edges,template,folder_ids[0],image_ids[0][7],7)
# In[] Seccionamiento por medio de distancia euclideana
#I_gray = rgb2gray(I)
#I_new = np.zeros([636,944])
#proms = list()
#r = 130
#for k in range(0,6):
#    centro = coordinates[k][:]
#    for i in range(0,635):
#        for j in range(0,943):
#            d = np.sqrt( abs( (centro[0] - i)**2 + (centro[1] - j)**2 ) ) 
#            if d <= r:
#                I_new[i,j] = I_gray[i,j]
#            
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.axis('off')
#ax1.imshow(I_new,cmap='gray')
 
 #-Entradas: Imagen en escala de grises I_gray, coordenadas de los centros
 #coordinates, las dimensiones más pequeñas dim como una lista de 2, y el radio.
#@@ -134,44 +141,66 @@ 
#def corr2d(I_gray,t_name,folder,name,ind):
# def pozos(I_gray,coordinates,r):
def pozos(I_gray,centro,r):
    h,w = I_gray.shape
    I_new = np.zeros([h,w])
#    for k in range(0,len(coordinates)):
#    centro = coordinates[k][:]
    for i in range(0,h):
        for j in range(0,w):
            d = np.sqrt( abs( (centro[1] - i)**2 + (centro[0] - j)**2 ) ) 
            if d <= r:
                I_new[i,j] = I_gray[i,j]
#    r = 130
#    for i in range(0,h):
#        for j in range(0,w):
#            x = centro[1] - i
#            y = centro[0] - j
#            d  = np.sqrt(abs( np.square(x) + np.square(y) ))
#            if any(d <= r):
#                I_new[i,j] = I_gray[i,j]
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax1.axis('off')
#    ax1.imshow(I_new,cmap='gray')
    return I_new

#h,w = I_gray.shape
#I_new = np.zeros([h,w])
#r = 130
#for i in range(0,h):
#    for j in range(0,w):
#        x = coord[:,0] - i
#        y = coord[:,1] - j
#        d  = np.sqrt(abs( np.square(x) + np.square(y) ))
#        if any(d <= r):
#            I_new[i,j] = I_gray[i,j]
            
            
#list comprehension

#>>> [(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]

#d  = np.sqrt(abs(centro-dim))
#I_gray = resize(I_gray,dim)
#I_new = pozos(I_gray,coord,r)


# In[] Promedio de ensambles
#Mínima dimensión de las imágenes (CORREGIR)
min_h = 606
min_w = 914

I_ensamble = np.zeros([606,914])
#for i in range(0,len(folder_ids)):
for j in range(0,len(image_ids[1][:])):
    PATH = gen + folder_ids[1] + '/' + image_ids[1][j]
    I = imread(PATH)
    I_gray = rgb2gray(I)
    I_gray = resize(I_gray,(min_h,min_w))        
    I_ensamble = I_ensamble + I_gray
    
I_ensamble = I_ensamble/28
I_new = pozos(I_gray,coord,r)
 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.axis('off')
ax1.imshow(I_ensamble,cmap='gray')
ax1.imshow(I_new,cmap='gray')
 
print(np.mean(I_ensamble[:]))
 
scipy.misc.imsave('Resultados/Promedio1.png' ,I_ensamble)
# In[]

#r = 50
def circle(r):
    dim = r*2 + 10
    Nigerrimo = np.zeros([dim,dim])
    centro = [dim/2,dim/2]
    for i in range(0,dim):
        for j in range(0,dim):
            d = np.sqrt(abs( (centro[0] - i)**2 + (centro[1] - j)**2 ))
            if d > r:
                Nigerrimo[i,j] = 1
    return Nigerrimo

#c = circle(r)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.axis('off')
#ax.imshow(c,cmap='gray')
