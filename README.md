# Proyecto-individual-2
Henry Data Science - Proyecto Individual Machine Learning

Información relevante

 Mercado inmobiliario
 
  Dentro de la sociedad globalizada e industrializada, es sabido que los precios de los inmuebles han presentado un constante cambio, por lo que quienes deseen invertir o vender una propiedad se enfrentan al fenómeno especulativo existente en la valorización de éstos. Esto, debido a la constante tendencia de las ciudades a crecer demográfica y comercialmente, llegando a un punto en donde no se tiene certeza de la valorización real dentro del sector en donde se desee invertir. Pese a que el precio depende, en cierta medida, de las tendencias que esté teniendo el mercado inmobiliario en un determinado tiempo, poder estimar adecuadamente el valor de una propiedad es una referencia clave para entender si es una buena
oportunidad, ya sea de compra o de venta.

Descripción del problema

  Una importante empresa inversora dentro del rubro de la inmobiliaria en Colombia, con el fin de que implemente un modelo de clasificación que permita clasificar el precio de las propiedades en venta, utilizando los datos que se han puesto a su disposición correspondientes al año 2020.   Para esto, específicamente, debe predecir la categorización de las propiedades entre baratas o caras, considerando como criterio el valor promedio de los precios
(la media). 

Entrega

  Deben tener el código en un script .py o Jupyter Notebook .ipynb, el cual debe incluir un buen EDA, feature engineerging y, de ser posible, un pipeline de Machine Learning para el procesamiento de datos que consideren necesario. Es importante explicar claramente cada paso realizado mediante comentarios en el script o textos formato markdown dentro del Notebook, pensar que cualquier persona (en este caso serán los Henry Mentors evaluadores) debe entender de la mejor manera posible cada razonamiento y pasos aplicados. Recuerden, además, que deben enviar el repositorio que contenga el proyecto, por lo que es importante que le dediquen tiempo también a esta parte, dejando todo ordenado y con un README acorde, que sirva de introducción al contenido dentro de éste.   Por otro lado, es obligatorio que el script genere un archivo .csv sólo con las predicciones, teniendo únicamente una sola columna (sin index) que debe llamarse 'pred' y tenga todos los valores de las predicciones, con un valor por fila. De no llamarse así la única columna, nuestro script de validación NO LO VA A TOMAR y no aparecerán en el dashboard. El nombre del archivo debe ser su usuario de GitHub, si su usuario de GitHub es 'pjr95', el archivo .csv con las predicciones debe llamarse 'pjr95.csv'. Vamos a validar tanto los datos que suban como el código, por lo que seguir estos pasos es fundamental.   Cuando entreguen les pedimos que verifiquen que su usuario de GitHub aparezca en el dashboard. En caso de que no aparezca, tal como se comentó más arriba, es debido a que el archivo entregado con las predicciones no cumple con los requisitos solicitados.
  
  Métrica a utilizar
  
  Como método de evaluación del desempeño del modelo, se utilizará la métrica de Exhaustividad (Recall) para las propiedades caras, a partir de la matriz de confusión (Confusion Matrix).   $$ Recall=\frac{TP} {TP+FN}$$   Donde $TP$ son los verdaderos positivos y $FN$ los falsos negativos. Adicionalmente, se incluye la Accuracy como métrica de control.  
  
  Archivos provistos
  
  Se proveen los archivos dentro del archivo comprimido 'properties_colombia.zip': 
  
'properties_colombia_train.csv': Contiene 197549 registros y 26 dimensiones, el cual incluye la información numérica del precio.
'propiedades_colombia_test.csv': Contiene 65850 registros y 25 dimensiones, el cual no incluye la información del precio.  

Descripción de las dimensiones

id - Identificador del aviso. No es único: si el aviso es actualizado por la inmobiliaria (nueva versión del aviso) se crea un nuevo registro con la misma id pero distintas fechas: de alta y de baja.
ad_type - Tipo de aviso (Propiedad, Desarrollo/Proyecto).
start_date - Fecha de alta del aviso.
end_date - Fecha de baja del aviso.
created_on - Fecha de alta de la primera versión del aviso.
lat - Latitud.
lon - Longitud.
l1 - Nivel administrativo 1: país.
l2 - Nivel administrativo 2: usualmente provincia.
l3 - Nivel administrativo 3: usualmente ciudad.
l4 - Nivel administrativo 4: usualmente barrio.
l5 - Nivel administrativo 5.
l6 - Nivel administrativo 6.
rooms - Cantidad de ambientes.
bedrooms - Cantidad de dormitorios (útil en el resto de los países).
bathrooms - Cantidad de baños.
surface_total - Superficie total en m².
surface_covered - Superficie cubierta en m².
price - Precio publicado en el anuncio.
currency - Moneda del precio publicado.
price_period - Periodo del precio (Diario, Semanal, Mensual)
title - Título del anuncio.
description - Descripción del anuncio.
property_type - Tipo de propiedad (Casa, Departamento, PH).
operation_type - Tipo de operación (Venta).
geometry - Puntos geométricos formados por las coordenadas latitud y longitud.  

1 - Data Cleaning

  Comprobación y toma de decisiones sobre valores nulos, errores de entrada o lectura.
  - Media
    Valores nulos convertidos a la media (promedio) de la columna para todas las características discretas o continuas.
  - String (NA)
    Se convierten los valores nulos en una cadena (texto) "NA" para todas las funciones nominales u ordinales.
  - 
    
  
          
2 - Exploración

  Analizar y visualizar los datos para ver y comprender mejor las relaciones y distribuciones.
  
  ![image](https://user-images.githubusercontent.com/43472426/199983307-4e9d8c04-67a0-4f7f-97b6-0b2ccdb3699d.png)
  
  ![image](https://user-images.githubusercontent.com/43472426/199985643-930d242d-074e-4480-95c4-9cafa55441d2.png)

   El conjunto de datos está claramente desbalanceado.No lo tomaremos en cuenta por ahora.

3 - Modelado y feature engineering.
   Elegir el mejor modelo y métodos de feature engineering.
     Se decidío que el mejor curso de acción era sacrificar la interpretabilidad por el rendimiento del modelo. Se utilizo PyCaret, para generar un            modelo.
     El setup de PyCaret es la función más importante, aquí es donde realizamos todos nuestros pasos del preprocesamiento de datos. 
      Parametros :
      • Data : Datos para el modelado 
      • Target : Columna de destino que queremos predecir en este caso si es cara o no
      • Session_id : ID de sesión definida por el usuario 
      • Normalize : Los modelos de aprendizaje automático funcionan bien cuando las características de entrada no tienen una gran variación,nuestro               dataset presenta diferentes escalas en alguna columnas. Es importante escalar entonces, por lo tanto, usamos el parámetro de normalización 
      • Transformation : Mientras que la normalización reduce, la transformación de varianza cambia los datos para que puedan representarse en una                 distribución gaussiana (curva normal). 
      • remove_multicollinearity : - Cuando los datos están altamente correlacionados, nuestros algoritmos tienden a no generalizar muy bien, por lo que           es importante eliminar la multicolinealidad usando los parámetros remove_multicolinealidad y multicolinealidad_umbral en la configuración. 
     
     
![image](https://user-images.githubusercontent.com/43472426/199996575-84722fc8-4b26-4105-899f-3947af978a2d.png)

![image](https://user-images.githubusercontent.com/43472426/199996907-37e3c3f2-00c1-45fd-b41e-dbc9dae4e690.png)

4 - Resultados.
    Comparativa con los ditintos modelos ejecutados
   ![image](https://user-images.githubusercontent.com/43472426/199998522-5155c306-4632-439b-8160-7768d3f4acca.png)
   ![image](https://user-images.githubusercontent.com/43472426/199998852-29cc6c21-ceaa-48d8-bcac-d90850f7383e.png)
   
    A los efectos de completar esta primera entrega se decidio tomar el modelo de Logistic Regression y realizarle ajustes por default.
    
   ![image](https://user-images.githubusercontent.com/43472426/200002460-23bf7ec4-309a-48dd-a437-e1103f6a9664.png)
   
   ![image](https://user-images.githubusercontent.com/43472426/200002794-f8e926a0-86d8-4df5-8d11-87c8785c0db9.png)

   Salida final contra el archivo de testeo ( properties_colombia_test.csv )
   
   ![image](https://user-images.githubusercontent.com/43472426/200003337-28eb36b0-ba47-445e-9a55-fb685ee87b04.png)




  
  

