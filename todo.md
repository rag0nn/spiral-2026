1#
    ~~1. spiral visualziton hallet.~~

2#
    1. ~~spidata kurulacak ve visualiztion ile birlikte check edilecek~~. ~~(Mevcut durumda hatalar var gibi görünüyor)~~.
    ~~2025 verileri ile tamamen yapı hazırlanacak ve şimdilik bırakılacak~~

3#
    ~~1. spiral training kısmına geçilecek.~~

4# ~~model klasik olarak backbone + FPNPAN + od olarak tanımlanacak,~~

5# ~~temel modelin üstüne time layer ve pos head mimarisi eklenecek~~

6# ~~pos ve od ayrı eğitim sağlanacak.~~

7# ~~veri seti tamamıyla hazır hale getirilecek~~

+# ~~Training için sub image tiling eklendi.~~

+# ~~Diğer veri seti kontrol edilecek çözünürlük iyiyse direkt olarak o veriler eklenecek~~

+# Model mimarisi tekrardan gözden geçirilecek, 640x640'dan vazgeçilebilir mi ona bakılacak zira veri çoğaltmada problem yaşanabiliyor.

+# Model eğitimi od sağlanacak.

+# Pos verileri ayarlanacak

8# model eğitimi pos sağlanacak

9# image search head'inden önce superpoint gibi veya direkt Akaze gibi extractor & Matcher eklenerek sağlanacak. Sonrasında mevcut modele bir image search head eklenecek

10# inference'da tiling kısmı eklenmeli eğer model inference'ta tile kullanılacaksa pos için bir farklılık sağlanmalı

11# Trajectory absolute istendiği için orantısal hesaplayıcı eklenmeli.

12# Kesinlikle bir SLAM aracı implemente edilmeli.

13# Object matcher head eklenmeli

14# Search ımage head eklenmeli

