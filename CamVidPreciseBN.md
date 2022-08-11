|    |miou(test)|
|----|----|
|BN(city)|56.3|
|BN(a2d2)|61.1|
|Precise bn(12 batch size; 2500 iter)|51.7|
|Precise bn(12 batch size; 10000 iter)|51.6|
|Precise bn(8 batch size; 2500 iter)|51.4|
|Precise bn(2 batch size; 2500 iter)|47.6|
|Precise bn(12 batch size; 2500 iter; no augment)|52.6|
|Precise bn(100 batch size; 50 iter; no augment)|52.9|
|Precise bn(40 batch size on detail and segment branch; 1000 iter; no augment)|55.0|
|Precise bn(40 batch size on detail and segment branch, 30 batch size on bga(independent); 1000 iter; no augment)|55.0|
|Precise bn(100 batch size base on city bn; 20 iter; no augment)|59.4|
|Precise bn(100 batch size base on city bn; 50 iter; no augment)|59.4|
|Precise bn(100 batch size base on city bn; 20 iter; no augment; only test)|59.5|
|Precise bn(100 batch size base on a2d2 bn; 20 iter; no augment; only test)|53.8|
|BN(city; segment head a2d2)|53.1|
|BN(a2d2; segment head a2d2)|61.8|
|Precise bn(100 batch size on a2d2 bn; segment head a2d2; 20 iter; no augment)|54.5|
|Precise bn(100 batch size on a2d2 bn; segment head a2d2; 10 iter; no augment; only test)|57.2|
|Precise bn(100 batch size on city bn; segment head a2d2; 20 iter; no augment)|57.5|
|Precise bn(100 batch size on city bn; segment head a2d2; 10 iter; no augment; only test)|58.8|

|    |Seg Head|evaluate dataset|miou(test)|
|----|----|----|----|
|BN(a2d2)|a2d2|a2d2|57.2|
|Precise BN(a2d2)|a2d2|a2d2|57.0|
|BN(city)|a2d2|a2d2|18.7|
|Precise BN(city)|a2d2|a2d2|37.6|
|BN(city)|city|city|72.4|
|Precise BN(city)|city|city|72.5|
|BN(a2d2)|city|city|56.9|
|Precise BN(a2d2)|city|city|59.3|

## the same affine layer
|    |Seg Head|evaluate dataset|miou(test)|
|----|----|----|----|
|Precise BN|a2d2|a2d2|55.8|
|BN|a2d2|a2d2|61.2|
