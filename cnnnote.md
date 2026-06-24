# Geçmişten Bugüne Evrişimli Sinir Ağlarının Mimari Gelişmeleri

Temelde özellik çıkarma işlevi gören CNN yapısı, zamanla farklı tür kullanım tarzlarıyla hem boyutsal eşitleme hem işlem maliyet minimizasyonu hem de farklı tür öğrenme şekilleri doğrultusunda yeni değişik mimarilere sahiptir. 

Bu yapılar bilinen konvolüsyon işlemini (veyahut conv katmanı da denebilir) değişik şekilde kombine etmeleri yoluyla farklılaşmaktadırlar, esasında konvolüsyon işleminin özünü değiştirmemektedirler.

Bu yazı da verilen örnekler pytorch temsilleri ile gerçekleştirilecektir.

### Ön Not
Öncelikle şahsımca da zaman zaman karıştırdığım, tensörün `conv`'dan çıktıktan sonra shape'nin değişmesinin önündeki şu düzeltmeyi hatırlatalım. 
Pytorchta `conv` katmanı şu şekilde tanımlanır:
```
nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size)
```
Padding'te eklendiği takdirde, bu işlem kanal boyutunda değişiklik yapar. Yani örneğin 3x100x100 olacak şekilde bir image tensor'ümüz olsun. Bu tensorü `nn.Conv2d(3, 10, kernel_size=3, padding=1) 
` katmanından geçirdiğimde 10x100x100 olacak şekilde yeni bir özellik tensörü elde ederiz. 

Bu tabiki bir sürpriz sayılmaz. Burada dikkat edilmesi gereken şudur: Bu katmanda tanımladığımız kernel_size=3 veya bazen kernel_size(3,3) olarakta tanımlanır. Esasında işlem sırasında (3,3,output_size) olarak işlev görür. Yani her bir kernel konvolüsyonu yapıldığında 100x100'ün bir 3x3'lük alanı yanında onun arkasında bulunan diğer boyuttaki öznitelikler hesaba katılır. Bu örnekte her bir kernel hesaplaması 3x3x3 yani 27 adet değerin işlemden geçirilmesi ile hesaplanır. Bu uyarı bir sonraki  bölümleri anlamada önemli olacaktır.

#### FLOP
Bir float değer için temel dört işlem sayısını ifade eder. Örneğin 3x3 conv layerda her output için 9 çarpma 8 toplama yapılır. Bu 17 FLOP olarak tanımlanır.

#### GFLOP
1 GFLOP = 10 Milyar FLOP
1 GFLOP = 1,000,000,000 işlem

### Problem Ne?
CNN'lerde tek "en iyi" mimari yoktur; farklı mimariler farklı sorunları hedefler. Temel sorunlar: hesaplama maliyeti, doğruluk–verim takası, derinlik sorunları (vanishing/exploding gradyan, degradation), çok ölçekli tespit ve gerçek zamanlı gereksinimler. Aşağıdaki alt başlıklar bu sorunları tanıtır ve sonraki mimarilerin neyi çözdüğü bağlamını kurar.

#### Implementasyon
FLOP tek başına pratik hızı göstermez; gerçek maliyeti bellek bant genişliği, kernel launch overhead ve paralelleştirme verimi belirler. Conv genelde im2col + GEMM ile uygulanır. Pratik hız için operatör füzyonu (Conv+BN+ReLU), nicemleme (INT8), pruning ve re-parameterization (YOLOv6'da) kullanılır.

#### Doğruluk & Hız Takası
Daha derin/geniş ağ → daha iyi doğruluk ama daha yavaş. İyi mimari, hedef deploy senaryosundaki Pareto frontier'deki en iyi noktadır. Scaling — width, depth, resolution — bu takası kontrol eder (edge: MobileNet, sunucu: ResNet/ConvNext). ResNet, degradation sorununu çözerek frontier'i kaydırır.

#### Kaybolan (Vanishing) ve Explode (Patlayan) Gradyan
Derin ağlarda backprop'ta zincirleme türevler katmanlar boyunca çarpılır. Vanishing'de türevler küçülüp erken katmanların öğrenmesini engeller; exploding'de büyüyüp eğitimi kararsızlaştırır. Çözümler: ReLU (AlexNet'le yaygınlaştı), He/Xavier başlatma, BatchNorm, gradient clipping ve özellikle skip connections (ResNet — ileride detaylanacak).


# Konvolüsyon Türleri: Spatial, Pointwise, Depthwise
### Spatial Conv (3x3,5x5,...)
Ön notta detaylı şekilde açıklanan en temel konvolüsyon şeklidir. Bu konvolüsyon her bir kernel için width, height ve channel boyutlarında hesaplama yapar. Yani bir kernel esasında 3 boyuta sahiptir denebilir. Bu genellikle width ve height'ın bir dilimi ve channel katmanının tamamının içeren bir kerneldir.

### PointWise Conv (1x1) Kanal Karıştırma
Kernel tanımı itibariyle 1x1'dir. Bu da 1x1xChannel'dır boyutunda bir kernele denk gelir. Bu kernel width ve height boyutunda hiç bir değişikliklik yapmaz iken kanallar arasında **kanal karıştırma** işlemi yapar. Yani taşıdığı bilgi öğenin bir bölgesine ait değildir. Sadece derinliği ile ilgilenir. 

Bu işlem şöyle tanımlanabilir:
```
nn.conv2d(3,100,kernel_size=(1,1))
```
Bu örnek tanımda her x,y değerinde 3 uzunluğundaki vektör için 100 uzunluğunda bir vektör oluşturulur.
Ve sonucunda (3,a,a)'dan (100,a,a) boyutunda tensör oluşturulur.

Bu işlemin yan avantajlarından biri diğer boyutları değiştirmeden boyutsal denkliği kolayca ayarlamaya yaramasıdır.  

Bu yapı bir yandan da fully connected nöron yapısını andırmaktadır.

### Depthwise Conv (a,a,1) Kanal Başına Çıkarım

Bu konvolüsyon türünde de 1x1xc kernellerin aksine kanallar birbirlerine karıştırılmaz. Her bir kanal kendi bölgesel çıkarımlarında bulunur. Bu işlem sonucunda da channel boyutunda bir değişim yaşanmamış olur.

### Sadede Gelelim
Bu farklı matematiksel `conv` katmanları CNN'lerin gelişiminde farklı kombinasyonlarla fark yaratmışlardır. Özellikle Spatial Conv yerine depthwise + pointwise konvolüsyonunun ardı ardına kullanılması bile sezgisel olarak daha ucuz olacağını bize hissettirir ki bu XCeption ve Mobilenet mimarilerinin temel taşlarını oluşturacaktır.

Bunlara ek olarak yatay (Nx1xC) ve dikey (1xNxC) konvolüsyonlar da bazı mimarilerde gündeme gelmiştir.

# Önemli Temel Mimariler (Temel Omurgalardan Bazıları)

### Alexnet
Probleme tamamen kafadan dalan bir mimaridir. Yaklaşımı basit ve düzdür. Yüksek kernel boyutlarına sahip sıralı conv katmanlarından sonuca giden basit bir yapıdır. ReLU aktivasyonunu yaygınlatırmıştır.

### VGGNet
Alexnet'e kıyasla daha düzenli ve daha ucuz olmaya çalışmış ve sadece 3x3 kernellerin peşi sıra hesaplanmasını kullanan mimaridir.

### Inception V1 (GoogleNet)
Bu mimari temel konvolüsyon yapısında her bir özellikle çıkarma için bir conv işleminden ziyade paralel conv işlemlerini içeren conv bloklarını kullanır. 

Her bir girdi için:
- 1x1 Conv
- 1X1 Conv(Boyut Azaltma) + 3x3 Conv
- 1x1 Conv(Boyut Azaltma) + 5x5 Conv
- 3x3 Maxpool + 1x1 Conv
 
katmanları uygulanır, devamında bu girdiler bir tensor olarak birleştirilir ve hatta genel bir spital conv layerden geçirilir. Bu bloktaki amaç birçok farklı düzeydeki özellikleri yakalayebilmektir.

### Inception V2
V1'deki 5x5'in getirdiği maliyet yükü sebebiyle 5x5 yerine 2 ardışık 3x3 katmanını getirir.

### Inception V3
Öncekilerden farklı olarak asimetrik conv'lar ekler.
- 1x1 Conv
- 1x1 Conv + 3x3 Conv
- 1x1 Conv + **3x1 Conv + 1x3 Conv**
- 1x1 Conv + 3X3 MaxPool

### XCeption 
Inceptionun aksine farklı branchlere sahip bir conv bloğunu sunmaz. Bunun yerine:
- (n,n,1) Depthwise conv + (1,1,c) Pointwise Conv

kullanılmasını önerir.

### ResNet
Resnet blokları isiminden de gelen residual (kalıntı) bloklarından oluşur. Bu mimarinin alâmetifarikası conv katmanlarını değiştirmek değildir. Bu sebeple de önemli bir kırılım noktası sayılabilir. bottleneck bloğu `1x1 (kanal azaltma) + 3x3 + 1x1 (kanal çoğaltma) + skip` şeklindedir.

Önceki modellerde derinlik arttıkça doğruluk düşüyordu (degradiation problem). Resnet bu sorunu çözmek için 'skip connection' yapısını önerir. Bu sayede de daha derin bir ağın geliştirilebileceğini öngörür.

#### Skip Connection
Önceki anlattığım modellerde veri akışı x ---> y şeklindeydi. Resnette veri şu şekilde akar:
x ---> y -> x + y   . Yani bu yapı girdi tensörü ile 'f(x) = y' çıktı tensörünü endeks bazlı olarak (x[i,j,k] + y[i,j,k]) değer toplamından geçirir. Bu yapı teorik olarak "a girişini b çıkışına nasıl dönüştürürümden" ziyade "a girişine ne eklersem b girişine ulaşırım?"'ı hesaplar. Yani aradaki dönüşümü ölçmeye çalışır. Bu da vanishing gradient (kaybolan gradyan) probleminin önüne geçer.

Tabiki bunun sağlanması içinde giriş ve çıkış tensörlerinin boyutsal olarak eşit olması gerekmektedir. Bu sebeple `1x1(kanal azaltma) Conv + 3x3 Conv + 1x1(Kanal Çoğaltma) + Skip connection` bloğundan oluşmaktadır.

### DenseNet
Özellikler katman katman unutulur ve bilgi kaybolur varsayımıyla ortaya çıkmıştır. Bilgi kaybını en aza indirmek için ResNet gibi skip connection yapar. Resnet skpi connection'u toplamayla yaparken bu model concat ile yapar.
Her adımda çıktı ile bir önceki adımın çıktısını concat eder.
```
x₁ = H₁(x₀)
x₂ = H₂([x₀, x₁])
x₃ = H₃([x₀, x₁, x₂])
```

### MobileNet
Tamamen hız'a ve parametre sayısını en aza indirmeye çalışır.
XCeption'daki **depthwise + pointwise** yaklaşımı daha agresif şekilde optimize edilir ve tüm conv katmanları sadece bu yapıyı içerir.

**V1**: Depthwise + pointwise kullanılır

**V2**: Pointwise + depthwise + pointwise kullanır. Ve bu yapıyı residual bir şekilde yapar.
Resnette bilgi geniş -> dar -> geniş olarak akıyordu. Burada tam tersine dar -> geniş -> dar olarak akar. Skip connection'lar da dar tensörler arasında toplanır. Bu yapıya **inverted residual block** adı verilir.

**V3**: Conv'da yenilik dışında daha modern NAS ve Attention mekanizmaları ve SENet (squeeze and excitation) bloklarına dahil eder.

SENet (SE block) olarakta geçer. Öncül attention mekanizmalarından biridir. Özellik haritasından average pooling ve tam bağlı katmanlar yardımıyla bir squeeze vektörü üretilir. Sonra bu vektörden de excitation vektörü üretilir. Sonrasında özellik haritasına bu vektör çarpılarak eklenir ve bu sayede hangi özelliğin daha önemli olduğuna dair bir çıkarım sağlanmış olur.

### Darknet-53
Residual bloklardan oluşan bir yapıdır. YOLOv3'ün de omurgasıdır. Conv katmanları: `1x1 Conv + 3x3 Conv + skip connection`

### CSPDarknet
CSPNet adı verilen ağın darknet'e uyarlanmış halidir. CSP (Cross Stage Partial) fikri: `Feature map’i ikiye böl → bir kısmı derin conv’dan geçir → diğer kısmı bypass et → sonra birleştir` prensibine dayanır.

Buna göre özellik tensörü iki parçaya ayrılır. İlk olduğu gibi sonuçtaki yerini alır. İkinci parça `conv + residual + conv` olacak şekilde sonuçta yerini alır.

Buradaki mantık şuna dayanır: Özellik tensörlerinin her bölgesi aynı derecede temsil içermez. Yani zaten iyi bulunan bir temsili yeni conv layer'larından geçirme diyor.

CSPDarknet zaman içinde aynı fikrin farklı implemantasyonlarıyla karşımıza çıkmıştır.

#### C3
```
Branch1: conv + n x residual bottleneck + conv

Branch2: conv (shortcut)

Concat + conv
```

#### C2
C3'ün daha hafifletilmiş versiyonu
```
Branch1: conv + n x residual bottleneck 

Branch2: conv (shortcut)

Concat 
```

#### C3K
C3'te bottleneck 1x1 + 3x3 + residual olarak tanımlıydı. C3K2'de yeni olan şey sadece bunun bottleneckin parametreye bağlanmasıdır.

#### C2F

Bundan önceki CSP mimarisinde **residual** yaklaşım tercih edilmişti. Bu mimaride **dense**
bloklar tercih edilerek özelliklerin yeniden kullanımı sağlandı.
```
Input
  ↓ Conv
  ↓ Split features
     ├─ B1 → Conv
     ├─ B2 → Conv(B1 + input)
     ├─ B3 → Conv(B2 + input)
     ├─ B4 → Conv(B3 + input)
  ↓ Concat(B1, B2, B3, B4, input)
  ↓ Conv (fusion)
```

#### C3K2

C3K gibi adlandırılsa da esasında C2F mantığında çalışır. C2F'te özellikler bir öncekiyle concat edilip  bottleneckten geçiriliyordu. Burada aynı şekilde bir bottleneckten geçirme var fakat bu bottleneckler birer C3K bottlenecki

#### SPP ve SPPF
Bir özellik haritasına 5x5 9x9 13x13 gibi paralel pooling uygulanarak farklı scale'lerde bilgi yakalamaya SPP yapısı deniyordu. 

SPPF bu yapıyı hız açısından genişleten yapıdır. Farklı boyutlarda kerneller kullanmak yerine 3 adet sıralı 5x5 Pooling koyar. Finalde de ilk çıktı ara çıktılar ve son çıktıyı concat eder
```
x
│
├── y1 = MaxPool5(x)
├── y2 = MaxPool5(y1)
├── y3 = MaxPool5(y2)
```
Sonra:

```
Concat(
 x,
 y1,
 y2,
 y3
)
```

#### PSA ve C2PSA
Conv işlemi sadece komşu layerlara bakar, Selft Attention ise her bir feature hücresini diğer feature hücresi ile ilişkilendirdiği için daha bağlantısal sonuçlar üretebilir.

**PSA**: 80x80 feature map için self attention 6400 değişken içerir ve bu çok pahalıdır. Bu sebeple PSA(Partial Self Attantion) mekanizmasında CSP'den biraz ayrışılarak özellik haritası ikiye bölünür. İlk branch self attentiondan geçirilir. İkinci branch CNN bottleneckten geçirilir.

**C2PSA**: PSA'nın CSP mantığıyla paketlenmiş halidir. Özellik haritasından ilki PSA'ya gönderilir. İkincisi bypass edilir. Sonra concat ile birleştirilir.

## Özet
Modern olarak bunların kombinasyonlarıyla kullanılan bloklar
```
C3K2
  ↓
Feature extraction

SPPF
  ↓
Multi-scale context

C2PSA
  ↓
Global context
```

# Boyun Yapıları

### FPN

Bir backbone'un sıralı conv blokları içerisindeki ara özellik haritaları alınır (C1,C2,C3,C4,C5). Bunlar 1x1 conv'dan geçirilerek channel boyutu eşitlenir. Sondan başlanılarak P5 = C5, P4 = C4 + UPSAMPLE(P5) vb. olarak eklenir (elementwise). 

Upsample bir çeşit interpolasyondur.
Ortaya çıkan özellik haritası farklı boyutlarda bilgileri yakalamaya yarar.

Alametifarikası sondan başa doğru ilerlemesidir.


### PAN

FPN'nin tek yönlü bilgi akışı varsayımıyla FPN'nin üstüne eklenen iki yönlü çözüm yaklaşımıdır.

FPN'nin üstüne bir de baştan sona bir akış ekler. Genelde özellik haritalarındaki downsampling conv işlemleri ile yapılır. Ve elementwise addition yerine genelde concat kullanılır.

FPN (top-down)
``` 
C5 → P5
C4 → P4
C3 → P3
C2 → P2
```
PAN (bottom-up path)
```
P2 → N3
P3 → N4
P4 → N5
```

### BIFPN

FPN/PAN yaklaşımı her özellik layerinın öneminin eşit varsayar.

BIFPN `P = w1*A + w2*B` olarak öğrenilebilir ağırlıklandırılarak bazı featurelara daha fazla önem verilir.

# Head Yapıları

### Anchor Tabanlı

Anchor her bir özellik haritası hücresi başına üretilen öntanımlı kutulara denir. Örneğin yolo modellerine her hücre başına 32x32 64x64 128x128 olarak anchorlar tanımlanırdı. 

Hesaplama verilen nesnenin merkez noktasına dayanandırılır. Objenin merkez noktasının bulunduğu hücrede yukarı da tanımlanan 3 anchor için IoU değerlerine bakılır ve en yakın olanı seçilir. Örneğin 2. yani 64x64 seçilir. Sonrasında bu anchor için xi,yi,wk,hk değerlerini hesaplar. Bu değerler anchorun ne kadar öteleneceği ve ne kadar çarpıtalacağını belirler. Bu sayede herhangi dikdörtgensel orantıya sahip bir obje, anchorun ötelenmesi ve çarpıtılması ile tahmin edilir. Yani model bir nesne için ne kadar çarpıtma ve öteleme yapacağını öğrenir. Seçilmeyen diğer anchorlar ise background olarak işaretlenerek burada nesne olmadığı öğrenilir.

Mantıken 3 anchor box yerine tek anchorla da bu iş teorik olarak ilerletilebilir fakat çok küçük veya çok büyük nesneler olduğunda çarpıtma değerleri çok yüksek ve küsürata hassas olacağından böyle bir yöntem izlenmiş, yani en yakın olana en az çarpıtma yapalım denmiş.

Bu eğitim için böyleydi fakat tahmin biraz daha farklı işliyor. 

Tahminde her bir hücre için tüm anchorlardan bir obje tahmini çıkartılıyor. İşte mesela i,j'deki 1. anchor box'dan bbox + class + score. Aynı şekilde 2. ve 3. anchor box için de bbox + class + score üretiliyor. Sonra score'lara bakılarak bir tresholding yapılıyor. Hatta belki en sonunda NMS algoritmasında da ek eleme ve birleştirme yapılıyor.

Bu yapının problemli kısmı çok fazla negatif olmasıdır. Eğitim sırasında Anchorların %98'i seçilmez. Keza tahmin kısmında da bir sürü tahmin yapılıp eleme yapılır, bu da bir gereksiz bir yük sayılabilir.

Loss fonksiyonu ise şu şekilde kurulur. Pozitif anchor için;

    Objectness loss: Bu anchor'da obje var mı?: 0,1: BCE
    Box Regression loss: Bu anchoru GT'ye nasıl dönüştürürüm?: tx,ty,tw,th: L1
    CLassification loss: Bu nesne hangi sınıf?: [0,n]: crossentropy:

Eğitim sırasında loss sadece negatif anchorlar için sadece objectness loss dahil edilir. Pozitif anchorlar için bu üç loss değeri toplanarak dahil edilir.  

**Focal Loss**: object detection’da özellikle şu problemi çözmek için tasarlanmış bir loss’tur: Modelin gördüğü örneklerin %99’u arka plan (negatif), çok azı nesne (pozitif). Bu dengesizlik öğrenmeyi bozuyor. Bu sebeple denge parametrelerine sahiptir. Zor örneklere daha fazla odaklanmayı sağlar.

**DFL (Distribution Focal Loss)**: Bir bbox için bir sayı tahmin etmek yerine bir dizi sayının olasılığını tahmin ettirmektir. Yani i,j,w,h olan bir tahminde i değeri bir skaler değer olarak tahmin edilmek yerine [0,1,...,n] dizisi için skorlar üreten bir şekilde loss üretilir.

### Anchor Free
Bu yaklaşımda anchor box'lar yoktur. Ama assigment ile ilgili farklı bir yaklaşım sunmaz. Assigment'tan kasıt bir objeyi hangi hücrenin temsil edeceğidir. 

Bu yaklaşımda özellik haritasındaki hücre bir anchor var mı varsa anchoru ne kadar deforme etmeliyimi öğrenmez, direkt olarak axa'lık bir anchor tanımı da yoktur. Bunun yerine örneğin center assigment'lı bir modelde, objenin merkezinin olduğu hücre için objectness = 1 olarak atanır ve bu hücre direkt olarak i,j,w,h, class, probs gibi sonuç üretir.

Hücrenin bir objeyi temsil etmediği durumda objectness= 0 olur, bbox loss'a dahil edilmez, class ise background index olarak tanımlanır.

Tahmin kısımda her bir hücre bir nesne tahmin eder. Sonrasında tresholding'le obje sayısı indirgenir. 

#### Coupled / Decoupled
Coupled head de classification ve regression için aynı özellik haritası kullanılır.
Yani ağ tek seferde x,y,w,h,objectness, class_probs gibi bir vektör üretir.

Classification ve regression aslında farklı biçimde tahminler olduğu için decoupled head'de özellik haritası iki ayrı conv zincirinden geçirilir. Bir head'de class olasılıkları tahmin edilirken diğer head'de x,y,w,h,objectness vektörleri üretilir.

Bu modelin daha hızlı öğrenmesine ve loss'un daha yumuşak düşmesine yol açar, Ve iki branch birbirini etkilemez.

#### NMS Durumu
Bu iki yaklaşım sonrasında da Non Maximum Supreesion algoritması kullanılabilir. Daha yeni modeller bunun alternatifini de sunar. NMS kullanmadan objeleri tek bir objeye indirgemeyi öğrenme ile alakalı olarak DETR yapısı gibi. Bundan aşağıdaki RT-DETR kısmında bahsedilmesi planlanmaktadır.

# Daha gelişmiş Uçtan Uca Modeller
Bu alanda daha çok nesne tespitini uçtan uca çözümleyen mimariler ile ilgilineceğiz. Gelişimlerine bakıldığında yaklaşımlar şöyle sınıflandırılabilir: One-Stage ve TWo Stage. İki aşamalı yaklaşımlarda önce nesnenin olabiliği alanlar tespit edilip sonrasında nesne tespit edilir. One stage yaklaşımlar her özellik için birer obje adayı üretilir ve çıkışta filtrelenir.

### SSD
SSD farklı boyutlarda nesne tespitini çözümleme iddiasıyla ortaya atılmıştır. GÖrsel ölnce bir omurgadan geçirilir. Sonrasında sırasıyla conv layerlarından geçilir örneğin conv5 -> conv6 -> conv7 gibi. Aynı zaman da her bir ek conv'larından sonra bir de detection head vardır ve anchorları tahmin eder.

### R-CNN
R-CNN mimarileri iki aşamalı detection'un en iyi örneklerindendir.

#### Selective Search
R-CNN'i anlamak için önce selective search'i anlamamız gerekir. Selective search algoritması bir görseli çok küçük hücrelere böler. Sonra hiyerarşik olarak komşu hücreler ile benzerliklere bakarak birleştirme işlemi gerçekleştirir. Yani görsel üzerindeki olası nesneleri benzerlikler yardımıyla tahmin eder.

R-CNN önce lokalizasyon adımını halleder ki bu da selective search dediğimiz şeydir.

Sonraki Tahmin adımında ise bu olası bölgeler CNN özellikler çıkarılır.

Sonrasında her bir bölge için bir SVM sınıflandırıcı eğitilir.

R-CNN 2014'de ortaya çıkmış, fikren güzel fakat pratik olarak çok yavaş bir yaklaşımdır.

### Fast R-CNN
R-CNN'de her bölge CNN'den geçiriliyordu. Fast R-CNN ilk iki adımın yerini değiştir. Önce tüm görselden özellik çıkarımı yapılır sonrasında bu özellik haritasında selective search yapılır. Bunların sonucunda da bir SVM eğitmek yerine ROI Pooling denen bir katman çalıştırır. Selective search sonrası farklı boyutlardaki kutuları sabit sayıdaki vektörlere çevirmeye yarar. Bu ardıkış Pooling katmanlarıyla sağlar.

### Faster R-CNN
R-CNN'lerin ağır kalan selective search kısmını öğrenilebilir hale getirir. Önce görüntü resnet, vgg gibi bir omurgadan geçirilir. Sonra yukarı da bahsedilen Anchor-based yaklaşımıyla objeleri tahmin eder. Zaten bu modelin alameti farikası anchor-based tahmini CNN içinde öğrenilebilir parametrelere bağlamasıdır.

### Mask R-CNN
<!--  -->


### RetinaNet
Retinanet tek aşamalı bir mimaridir. `Omurga -> FPN -> Anchor-Based Head` olarak tahmin sağlar. Alametifarikası loss için FocalLoss'u önermesidir. Daha önce de bahsedilen çok falza negatif (background) anchor box olduğu için zor örneklerin öğrenilmesini kolaylaştırmaya yarar.

### YOLO
YOLO model ailesi modelleri birbirinden sadece mimari yaklaşımlarla ayrışmaz, Loss, veri çoğaltma eğitim süreçleri gibi daha pratik sebeplerden de birbirilerinden ayrışırlar.

#### V3
Omurga: Darknet-53
Baş: Anchor-based
Bu omurgadan farklı seviyelerde özellik haritaları alınır. Diyelim ki C1(52x52),C2(26x26),C3(13x13).
C3 alınır ve direkt detection head'e gönderilir. C2, C3 upsample edilir C2 ile concat edilir ve head'e gönderilir. C1, C2 upsample edilip concat edilip heade gönderilir.

#### V4
Omurga: CSPDarknet
Boyun: SPP+FPN+PAN
Baş: Anchor-based

#### V5
Omurga: CSPDarknet
Boyun: SPPF+FPN+PAN
Baş: Anchor Based

C3 blokları kullanılır.

Aynı zamanda biraz daha pratiğe optimize edilmiş yenilikler sunar.

#### V6
Omurga: EfficientRep
Boyun: Rep-PAN
Baş: Decoupled, anchor-base/anchor-free

EfficientRep'in temel fikri training zamanı yüksek parametre öğrenme tahmin zamanı düşük parametre geçmektir. Yani tahmin çok hızlı halledilir. Bunu "re-parameterization" fikriyle sağlar.

İçindeki blok:
```
x → Conv3x3
  → Conv1x1
  → Identity
  → sum
```
şeklindedir. Bu aslında önceki conv katmanlarına da benzer. Fakat inference zamanı geldiğinde model bu katmanı tek bir blokta matematiksel olarak birleştirerek daha hızlı çıktı sağlar. 

Nasıl mümkün oluyor: `Conv + BN + Conv` lineer işlemler, konvolüsyonların toplanması, batchhhormun sabit ağırlığa dönüştürülmesi yoluyla birleştirilebilirler. Bu 'RepConv' olarakta adlandırılır.

Aynı zamanda PAN katmanı da mimari olarak aynıdır sadece parametreleri yine RepConv'lar ile değiştirerek sıkıştırma yapar.

#### V7
Omurga: E-ELAN (Extended Efficient Layer Aggregation Network)
Boyun:	Rep-PAN	ELAN-based PAN
Baş:	decoupled	anchor-based + aux head

**Auxiliary Head**: Trainingte olan ama inference'te olmayan ekstra bir baş. Bu baş yapısı PAN sonrası orta büyüklükteki özellikleri yakalayan özellik haritasından sonra konur. Bu ekstra loss ekleyerek daha modelin derin katmanları daha iyi öğrenmesini sağlayan bir ekstra baştır. Loss `loss_total = loss_main + λ * loss_aux`.

**E-LAN blokları** bir özellik haritasını split eder, ayrı ayrı conv layerlarından geçirir, sonra concat eder. Burada bir path için conv layer sayısı değişkenlik gösterebilir.

Buna ek olarak RepConv (YOLOV6daki) katmanları da içerir.

**ELAN-based PAN** ise normal PAN çıkışından sonra ELAN bloklarıyla konvolüsyona devam edilmesidir.

#### V8
Omurga: C2F Tabanlı Omurga
Boyun: SPPF+FPN+PAN
Baş: Decoupled Anchor Free, DFL

Anchor free head'lere ve C2F bloklara geçilmiştir.

#### V11
Omurga: C3K2, C2PSA Tabanlı Omurga
Boyun: SPPF+FPN+PAN
Baş: Decoupled Anchor Free, DFL

Yeni C3K2, C2PSA tabanlı bir omurgaya geçilmiştir.

#### V13
Omurga: DS-C3k2+HyperACE
Boyun: SPPF+FPN+PAN
Baş: Decoupled Anchor Free, DFL

Ultralytics'ten ayrı akademik bir modeldir.

**DS-C3k2** normal C3K2 ama 1x1 ile FLOP düşürülüyor.

**FullPad**: Bazı özellikler omurga, boyun, baş kısımlarına dağıtılıyor.

**HyperACE**: Fikte göre PSA gibi katmanlar iki farklı özellikler haritasını bağlıyor. Bunun 2'den fazla katmanı birbirine bağlıyor.

#### V26
Omurga: Öncekilere benzer CSP blokların kombinasyonu
Boyun: FPN/PAN
Baş: Decoupled, Anchor-Free,DFL ve E2E assignment (NMS-free hedefi)

İçerdiği birkaç minik attention blokları dışında en önemli fark E2E(end to end) assignment ile NMS-free olmasıdır. Bu model hangi nesneyi seçeceğine kendi karar versin mantığıdır. 

Model eğitilrken çıkan sonuçlar GT'ler ile bir cost matrix yoluyla eşleştirilir. Eşleştirilen öğelerden çıkan cost'lara göre loss değeri hesaplanırken ekleme yapılır. Tahmin aşamasına hungarian algoritması yoktur sadece loss hesaplamak ve modeli objeye zorlamak için eklenir.

### Vision Transformer Mimarisi (ViT)
Hücreleri cnn yerine transformer ile işlemeyi önerir. Daha geniş perspektife bakar ama çok daha maliyetlidir ve çok daha fazla veriye ihtiyaç duyar.

### ConvNext
ViT mimarisi daha geniş receptive field ile başarı sağladığından şu fikir öne sürülür. Daha geniş bir reseptive field ile daha çok bağlantı sağlanacağığından 7x7 depthwise conv katmanları kullanır.

### Swin
Geniş bir attention yerine küçük windowlarda attention yapmayı önerir. Attention'daki O(N²) maliyeti CNN'deki O(N) maliyetine yaklaştırır. Ama aslında böyle yapınca yine global değil lokal bilgi öğrenilir.

**Shifted Window** global bilgiyi almak için bir sonraki layerda window kaydırılır. Bilgi yavaş yavaş tüm görüntüye yayılır.

### DETR (2020)
Nesne tespiti problemini "set prediction" problemine çevirir. Yani bu model ben direkt "nesneyi tespit ederim" der.

Backbone olarak klasik CNN backbone kullanır. Sonrasındaki özellik haritası attention'a tabi tutulur. Baş kısmında anchor free'dir ama YOLO gibi değildir çünkü box'lar attention query'leri ile oluşturulur. Bu sebeple eğitim sırasında hungarian algoritması kullanılır. Yani model bir image'tan kaç adet ve nasıl obje tahmin edeceğini öğrenir. Loss hungarian sonrası eşleştirilen nesneler üzerinden olur.

### RT-DETR 
DETR çok güçlü global reasoninge sahipti ama eğitim ve tahmin kısmında çok ağırdı. RT-DETR bu durumu daha verimli hale getirmeyi amaçlar. Bunun için ilk olarak backbone'u modern conv layerları ile değiştirir. Full global attention yerine düşürülmüş attention ile seçici özellik çıkarımı yapar yani her token her token'a bakmaz.
Bir de FPN/PAN gibi "multi scale fusion" yapısı eklenmiştir.

### LW-DETR 
Lightweight DETR, DETR mimarisini minimum maliyetli şekle dönüştürmeyi amaçlar.
1. Klasik backbone yerine efficientnet, mobilenet gibi backbonelar kullanır.
2. Daha hafif bir transformer encoder, her token her tokena bakmaz.
Bir manada RT-DETR'in daha da ucuz hale getirilmiş versiyonudur.

### D-FINE 
D-FINE, DETR tabanlı object detection yaklaşımını koruyup, özellikle bounding box localization kalitesini artırmak için multi-stage refinement ve daha hassas matching/loss tasarımları kullanan fine-grained detection yaklaşımıdır.
<!-- eklenecek -->