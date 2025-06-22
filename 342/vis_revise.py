import matplotlib.pyplot as plt
import re
import numpy as np

# Raw Mahalanobis distance data
raw_output_grouped_index = """
Line 1 => Mahalanobis distance = 590.358
Line 2 => Mahalanobis distance = 6110.228
Line 3 => Mahalanobis distance = 5959.549
Line 4 => Mahalanobis distance = 21984.062
Line 5 => Mahalanobis distance = 2864.312
Line 6 => Mahalanobis distance = 5396.019
Line 7 => Mahalanobis distance = 2843.905
Line 8 => Mahalanobis distance = 3764.919
Line 9 => Mahalanobis distance = 4081.013
Line 10 => Mahalanobis distance = 1394.127
Line 11 => Mahalanobis distance = 2853.186
Line 12 => Mahalanobis distance = 1364.206
Line 13 => Mahalanobis distance = 2369.161
Line 14 => Mahalanobis distance = 22756.696
Line 15 => Mahalanobis distance = 4689.899
Line 16 => Mahalanobis distance = 20510.142
Line 17 => Mahalanobis distance = 9762.561
Line 18 => Mahalanobis distance = 5806.022
Line 19 => Mahalanobis distance = 15690.841
Line 20 => Mahalanobis distance = 5232.039
Line 21 => Mahalanobis distance = 13816.143
Line 22 => Mahalanobis distance = 9452.447
Line 23 => Mahalanobis distance = 7827.927
Line 24 => Mahalanobis distance = 8731.823
Line 25 => Mahalanobis distance = 5299.436
Line 26 => Mahalanobis distance = 6017.618
Line 27 => Mahalanobis distance = 5280.968
Line 28 => Mahalanobis distance = 2768.413
Line 29 => Mahalanobis distance = 788.405
Line 30 => Mahalanobis distance = 2528.024
Line 31 => Mahalanobis distance = 744.019
Line 32 => Mahalanobis distance = 2299.295
Line 33 => Mahalanobis distance = 590.926
Line 34 => Mahalanobis distance = 2212.417
Line 35 => Mahalanobis distance = 928.553
Line 36 => Mahalanobis distance = 1892.534
Line 37 => Mahalanobis distance = 780.733
Line 38 => Mahalanobis distance = 1593.287
Line 39 => Mahalanobis distance = 998.204
Line 40 => Mahalanobis distance = 641.333
Line 41 => Mahalanobis distance = 1192.945
Line 42 => Mahalanobis distance = 1059.238
Line 43 => Mahalanobis distance = 2758.961
Line 44 => Mahalanobis distance = 766.852
Line 45 => Mahalanobis distance = 2533.845
Line 46 => Mahalanobis distance = 898.697
Line 47 => Mahalanobis distance = 2277.130
Line 48 => Mahalanobis distance = 901.450
Line 49 => Mahalanobis distance = 2013.374
Line 50 => Mahalanobis distance = 1141.097
Line 51 => Mahalanobis distance = 982.337
Line 52 => Mahalanobis distance = 927.354
Line 53 => Mahalanobis distance = 1550.626
Line 54 => Mahalanobis distance = 913.871
Line 55 => Mahalanobis distance = 1152.070
Line 56 => Mahalanobis distance = 1007.895
Line 57 => Mahalanobis distance = 844.830
Line 58 => Mahalanobis distance = 3695.943
Line 59 => Mahalanobis distance = 1212.619
Line 60 => Mahalanobis distance = 3411.480
Line 61 => Mahalanobis distance = 634.823
Line 62 => Mahalanobis distance = 3277.635
Line 63 => Mahalanobis distance = 616.404
Line 64 => Mahalanobis distance = 3248.512
Line 65 => Mahalanobis distance = 737.987
Line 66 => Mahalanobis distance = 3091.291
Line 67 => Mahalanobis distance = 3391.152
Line 68 => Mahalanobis distance = 3495.355
Line 69 => Mahalanobis distance = 1532.549
Line 70 => Mahalanobis distance = 608.762
Line 71 => Mahalanobis distance = 1416.631
Line 72 => Mahalanobis distance = 812.730
Line 73 => Mahalanobis distance = 1002.683
Line 74 => Mahalanobis distance = 839.986
Line 75 => Mahalanobis distance = 4057.390
Line 76 => Mahalanobis distance = 449.243
Line 77 => Mahalanobis distance = 4081.898
Line 78 => Mahalanobis distance = 5931.377
Line 79 => Mahalanobis distance = 6581.895
Line 80 => Mahalanobis distance = 1067.133
Line 81 => Mahalanobis distance = 652.546
Line 82 => Mahalanobis distance = 872.525
Line 83 => Mahalanobis distance = 442.043
Line 84 => Mahalanobis distance = 252.889
Line 85 => Mahalanobis distance = 799.215
Line 86 => Mahalanobis distance = 550.477
Line 87 => Mahalanobis distance = 522.756
Line 88 => Mahalanobis distance = 467.752
Line 89 => Mahalanobis distance = 1355.712
Line 90 => Mahalanobis distance = 518.740
Line 91 => Mahalanobis distance = 1227.166
Line 92 => Mahalanobis distance = 465.531
Line 93 => Mahalanobis distance = 1121.993
Line 94 => Mahalanobis distance = 247.164
Line 95 => Mahalanobis distance = 1140.417
Line 96 => Mahalanobis distance = 485.056
Line 97 => Mahalanobis distance = 259.336
Line 98 => Mahalanobis distance = 1013.769
Line 99 => Mahalanobis distance = 661.590
Line 100 => Mahalanobis distance = 788.094
Line 101 => Mahalanobis distance = 278.164
Line 102 => Mahalanobis distance = 870.317
Line 103 => Mahalanobis distance = 615.067
Line 104 => Mahalanobis distance = 569.562
Line 105 => Mahalanobis distance = 519.148
Line 106 => Mahalanobis distance = 1336.426
Line 107 => Mahalanobis distance = 636.826
Line 108 => Mahalanobis distance = 1174.913
Line 109 => Mahalanobis distance = 611.453
Line 110 => Mahalanobis distance = 1011.552
Line 111 => Mahalanobis distance = 246.024
Line 112 => Mahalanobis distance = 1054.507
Line 113 => Mahalanobis distance = 366.057
Line 114 => Mahalanobis distance = 950.037
Line 115 => Mahalanobis distance = 633.468
Line 116 => Mahalanobis distance = 712.057
Line 117 => Mahalanobis distance = 469.508
Line 118 => Mahalanobis distance = 340.123
Line 119 => Mahalanobis distance = 491.124
Line 120 => Mahalanobis distance = 431.931
Line 121 => Mahalanobis distance = 2351.450
Line 122 => Mahalanobis distance = 632.002
Line 123 => Mahalanobis distance = 2225.215
Line 124 => Mahalanobis distance = 493.267
Line 125 => Mahalanobis distance = 545.543
Line 126 => Mahalanobis distance = 2251.830
Line 127 => Mahalanobis distance = 643.986
Line 128 => Mahalanobis distance = 2137.089
Line 129 => Mahalanobis distance = 1934.106
Line 130 => Mahalanobis distance = 480.422
Line 131 => Mahalanobis distance = 289.822
Line 132 => Mahalanobis distance = 895.748
Line 133 => Mahalanobis distance = 686.212
Line 134 => Mahalanobis distance = 532.283
Line 135 => Mahalanobis distance = 478.261
Line 136 => Mahalanobis distance = 190.559
Line 137 => Mahalanobis distance = 116.577
Line 138 => Mahalanobis distance = 277.594
Line 139 => Mahalanobis distance = 225.723
Line 140 => Mahalanobis distance = 6854.539
Line 141 => Mahalanobis distance = 3570.563
Line 142 => Mahalanobis distance = 3148.259
Line 143 => Mahalanobis distance = 126.539
Line 144 => Mahalanobis distance = 66.545
Line 145 => Mahalanobis distance = 83.491
Line 146 => Mahalanobis distance = 70.634
Line 147 => Mahalanobis distance = 134.708
Line 148 => Mahalanobis distance = 62.471
Line 149 => Mahalanobis distance = 188.090
Line 150 => Mahalanobis distance = 53.151
Line 151 => Mahalanobis distance = 51.445
Line 152 => Mahalanobis distance = 125.019
Line 153 => Mahalanobis distance = 35.017
Line 154 => Mahalanobis distance = 7102.203
Line 155 => Mahalanobis distance = 6846.713
Line 156 => Mahalanobis distance = 132.826
Line 157 => Mahalanobis distance = 139.840
Line 158 => Mahalanobis distance = 151.586
Line 159 => Mahalanobis distance = 156.136
Line 160 => Mahalanobis distance = 157.487
Line 161 => Mahalanobis distance = 48.366
Line 162 => Mahalanobis distance = 67.800
Line 163 => Mahalanobis distance = 93.555
Line 164 => Mahalanobis distance = 300.064
Line 165 => Mahalanobis distance = 301.521
Line 166 => Mahalanobis distance = 380.397
Line 167 => Mahalanobis distance = 384.174
Line 168 => Mahalanobis distance = 391.737
Line 169 => Mahalanobis distance = 860.479
Line 170 => Mahalanobis distance = 864.107
Line 171 => Mahalanobis distance = 2144.497
Line 172 => Mahalanobis distance = 2184.462
Line 173 => Mahalanobis distance = 2260.562
Line 174 => Mahalanobis distance = 1174.292
Line 175 => Mahalanobis distance = 1179.523
Line 176 => Mahalanobis distance = 16.762
Line 177 => Mahalanobis distance = 33.739
Line 178 => Mahalanobis distance = 368.200
Line 179 => Mahalanobis distance = 723.439
Line 180 => Mahalanobis distance = 724.161
Line 181 => Mahalanobis distance = 477.328
Line 182 => Mahalanobis distance = 475.886
Line 183 => Mahalanobis distance = 53.720
Line 184 => Mahalanobis distance = 662.920
Line 185 => Mahalanobis distance = 661.001
Line 186 => Mahalanobis distance = 1756.531
Line 187 => Mahalanobis distance = 1844.881
Line 188 => Mahalanobis distance = 1073.849
Line 189 => Mahalanobis distance = 642.673
Line 190 => Mahalanobis distance = 640.337
Line 191 => Mahalanobis distance = 596.612
Line 192 => Mahalanobis distance = 602.113
Line 193 => Mahalanobis distance = 245.173
Line 194 => Mahalanobis distance = 155.711
Line 195 => Mahalanobis distance = 133.782
Line 196 => Mahalanobis distance = 747.711
Line 197 => Mahalanobis distance = 750.295
Line 198 => Mahalanobis distance = 231.827
Line 199 => Mahalanobis distance = 197.805
Line 200 => Mahalanobis distance = 185.073
Line 201 => Mahalanobis distance = 230.351
Line 202 => Mahalanobis distance = 230.941
Line 203 => Mahalanobis distance = 181.924
Line 204 => Mahalanobis distance = 633.440
Line 205 => Mahalanobis distance = 630.451
Line 206 => Mahalanobis distance = 2038.016
Line 207 => Mahalanobis distance = 2118.379
Line 208 => Mahalanobis distance = 712.053
Line 209 => Mahalanobis distance = 358.691
Line 210 => Mahalanobis distance = 357.524
Line 211 => Mahalanobis distance = 1002.169
Line 212 => Mahalanobis distance = 1005.642
Line 213 => Mahalanobis distance = 597.788
Line 214 => Mahalanobis distance = 257.969
Line 215 => Mahalanobis distance = 252.578
Line 216 => Mahalanobis distance = 1907.575
Line 217 => Mahalanobis distance = 1899.474
Line 218 => Mahalanobis distance = 991.019
Line 219 => Mahalanobis distance = 983.620
Line 220 => Mahalanobis distance = 980.451
Line 221 => Mahalanobis distance = 408.224
Line 222 => Mahalanobis distance = 408.331
Line 223 => Mahalanobis distance = 103.627
Line 224 => Mahalanobis distance = 71.478
Line 225 => Mahalanobis distance = 47.270
Line 226 => Mahalanobis distance = 111.617
Line 227 => Mahalanobis distance = 105.324
Line 228 => Mahalanobis distance = 498.979
Line 229 => Mahalanobis distance = 493.440
Line 230 => Mahalanobis distance = 491.811
Line 231 => Mahalanobis distance = 427.062
Line 232 => Mahalanobis distance = 421.436
Line 233 => Mahalanobis distance = 152.885
Line 234 => Mahalanobis distance = 116.595
Line 235 => Mahalanobis distance = 88.388
Line 236 => Mahalanobis distance = 128.522
Line 237 => Mahalanobis distance = 125.823
Line 238 => Mahalanobis distance = 413.006
Line 239 => Mahalanobis distance = 408.864
Line 240 => Mahalanobis distance = 406.724
Line 241 => Mahalanobis distance = 419.041
Line 242 => Mahalanobis distance = 418.991
Line 243 => Mahalanobis distance = 853.096
Line 244 => Mahalanobis distance = 851.935
Line 245 => Mahalanobis distance = 852.161
Line 246 => Mahalanobis distance = 1862.682
Line 247 => Mahalanobis distance = 1860.656
Line 248 => Mahalanobis distance = 2097.072
Line 249 => Mahalanobis distance = 1992.062
Line 250 => Mahalanobis distance = 1933.884
Line 251 => Mahalanobis distance = 131.807
Line 252 => Mahalanobis distance = 134.843
Line 253 => Mahalanobis distance = 461.223
Line 254 => Mahalanobis distance = 789.667
Line 255 => Mahalanobis distance = 790.344
Line 256 => Mahalanobis distance = 1993.858
Line 257 => Mahalanobis distance = 2084.916
Line 258 => Mahalanobis distance = 1029.094
Line 259 => Mahalanobis distance = 721.559
Line 260 => Mahalanobis distance = 720.225
Line 261 => Mahalanobis distance = 316.395
Line 262 => Mahalanobis distance = 318.316
Line 263 => Mahalanobis distance = 73.996
Line 264 => Mahalanobis distance = 500.695
Line 265 => Mahalanobis distance = 497.608
Line 266 => Mahalanobis distance = 102.949
Line 267 => Mahalanobis distance = 113.319
Line 268 => Mahalanobis distance = 351.766
Line 269 => Mahalanobis distance = 687.708
Line 270 => Mahalanobis distance = 687.601
Line 271 => Mahalanobis distance = 412.840
Line 272 => Mahalanobis distance = 411.155
Line 273 => Mahalanobis distance = 12.176
Line 274 => Mahalanobis distance = 393.670
Line 275 => Mahalanobis distance = 395.262
Line 276 => Mahalanobis distance = 1027.508
Line 277 => Mahalanobis distance = 1031.168
Line 278 => Mahalanobis distance = 243.654
Line 279 => Mahalanobis distance = 100.639
Line 280 => Mahalanobis distance = 88.248
Line 281 => Mahalanobis distance = 1234.922
Line 282 => Mahalanobis distance = 1232.575
Line 283 => Mahalanobis distance = 586.203
Line 284 => Mahalanobis distance = 92.215
Line 285 => Mahalanobis distance = 91.933
Line 286 => Mahalanobis distance = 128.325
Line 287 => Mahalanobis distance = 131.149
Line 288 => Mahalanobis distance = 295.326
Line 289 => Mahalanobis distance = 763.974
Line 290 => Mahalanobis distance = 759.741
Line 291 => Mahalanobis distance = 1453.068
Line 292 => Mahalanobis distance = 1511.377
Line 293 => Mahalanobis distance = 1139.797
Line 294 => Mahalanobis distance = 842.446
Line 295 => Mahalanobis distance = 828.839
Line 296 => Mahalanobis distance = 1176.230
Line 297 => Mahalanobis distance = 1175.242
Line 298 => Mahalanobis distance = 1177.278
Line 299 => Mahalanobis distance = 507.927
Line 300 => Mahalanobis distance = 509.745
Line 301 => Mahalanobis distance = 302.775
Line 302 => Mahalanobis distance = 308.163
Line 303 => Mahalanobis distance = 318.581
Line 304 => Mahalanobis distance = 39.851
Line 305 => Mahalanobis distance = 51.170
Line 306 => Mahalanobis distance = 239.687
Line 307 => Mahalanobis distance = 243.876
Line 308 => Mahalanobis distance = 251.277
Line 309 => Mahalanobis distance = 57.084
Line 310 => Mahalanobis distance = 70.211
"""

# Raw NLM data
raw_output_nlm = """
--- Maximum Normalized Lagrangian Multipliers per Line ---
Line 1 ('1'): Max NLM = 507.8
Line 2 ('2'): Max NLM = 5680
Line 3 ('3'): Max NLM = 5401
Line 4 ('4'): Max NLM = 1.93e+04
Line 5 ('5'): Max NLM = 2295
Line 6 ('6'): Max NLM = 4326
Line 7 ('7'): Max NLM = 2252
Line 8 ('8'): Max NLM = 3118
Line 9 ('9'): Max NLM = 2834
Line 10 ('10'): Max NLM = 1033
Line 11 ('11'): Max NLM = 2222
Line 12 ('12'): Max NLM = 997.3
Line 13 ('13'): Max NLM = 1783
Line 14 ('14'): Max NLM = 1.912e+04
Line 15 ('15'): Max NLM = 3328
Line 16 ('16'): Max NLM = 1.796e+04
Line 17 ('17'): Max NLM = 8358
Line 18 ('18'): Max NLM = 4785
Line 19 ('19'): Max NLM = 1.343e+04
Line 20 ('20'): Max NLM = 4332
Line 21 ('21'): Max NLM = 1.172e+04
Line 22 ('22'): Max NLM = 8134
Line 23 ('23'): Max NLM = 6798
Line 24 ('24'): Max NLM = 7072
Line 25 ('25'): Max NLM = 4342
Line 26 ('26'): Max NLM = 4801
Line 27 ('27'): Max NLM = 4290
Line 28 ('28'): Max NLM = 2220
Line 29 ('29'): Max NLM = 653.6
Line 30 ('30'): Max NLM = 2023
Line 31 ('31'): Max NLM = 613
Line 32 ('32'): Max NLM = 1841
Line 33 ('33'): Max NLM = 421.1
Line 34 ('34'): Max NLM = 1797
Line 35 ('35'): Max NLM = 762.2
Line 36 ('36'): Max NLM = 1535
Line 37 ('37'): Max NLM = 642.3
Line 38 ('38'): Max NLM = 1298
Line 39 ('39'): Max NLM = 802.9
Line 40 ('40'): Max NLM = 456.3
Line 41 ('41'): Max NLM = 935.8
Line 42 ('42'): Max NLM = 853.5
Line 43 ('43'): Max NLM = 2192
Line 44 ('44'): Max NLM = 628.8
Line 45 ('45'): Max NLM = 2015
Line 46 ('46'): Max NLM = 742.2
Line 47 ('47'): Max NLM = 1798
Line 48 ('48'): Max NLM = 747.8
Line 49 ('49'): Max NLM = 1572
Line 50 ('50'): Max NLM = 853.5
Line 51 ('51'): Max NLM = 790
Line 52 ('52'): Max NLM = 767.9
Line 53 ('53'): Max NLM = 1262
Line 54 ('54'): Max NLM = 744.9
Line 55 ('55'): Max NLM = 924.8
Line 56 ('56'): Max NLM = 824.7
Line 57 ('57'): Max NLM = 712.6
Line 58 ('58'): Max NLM = 3066
Line 59 ('59'): Max NLM = 966
Line 60 ('60'): Max NLM = 2829
Line 61 ('61'): Max NLM = 499.5
Line 62 ('62'): Max NLM = 2726
Line 63 ('63'): Max NLM = 440.6
Line 64 ('64'): Max NLM = 2691
Line 65 ('65'): Max NLM = 605.1
Line 66 ('66'): Max NLM = 2562
Line 67 ('67'): Max NLM = 2687
Line 68 ('68'): Max NLM = 2794
Line 69 ('69'): Max NLM = 1160
Line 70 ('70'): Max NLM = 433
Line 71 ('71'): Max NLM = 1138
Line 72 ('72'): Max NLM = 670.6
Line 73 ('73'): Max NLM = 804.3
Line 74 ('74'): Max NLM = 691.5
Line 75 ('75'): Max NLM = 2836
Line 76 ('76'): Max NLM = 331.4
Line 77 ('77'): Max NLM = 2845
Line 78 ('78'): Max NLM = 4075
Line 79 ('79'): Max NLM = 4559
Line 80 ('80'): Max NLM = 790.3
Line 81 ('81'): Max NLM = 469.5
Line 82 ('82'): Max NLM = 643.7
Line 83 ('83'): Max NLM = 335.7
Line 84 ('84'): Max NLM = 178.6
Line 85 ('85'): Max NLM = 550.9
Line 86 ('86'): Max NLM = 409
Line 87 ('87'): Max NLM = 358.9
Line 88 ('88'): Max NLM = 344.9
Line 89 ('89'): Max NLM = 1005
Line 90 ('90'): Max NLM = 387.2
Line 91 ('91'): Max NLM = 905.1
Line 92 ('92'): Max NLM = 338.4
Line 93 ('93'): Max NLM = 819.6
Line 94 ('94'): Max NLM = 172.8
Line 95 ('95'): Max NLM = 841.7
Line 96 ('96'): Max NLM = 374.6
Line 97 ('97'): Max NLM = 182.1
Line 98 ('98'): Max NLM = 732.6
Line 99 ('99'): Max NLM = 478.5
Line 100 ('100'): Max NLM = 563.8
Line 101 ('101'): Max NLM = 188.6
Line 102 ('102'): Max NLM = 592.9
Line 103 ('103'): Max NLM = 449
Line 104 ('104'): Max NLM = 387.8
Line 105 ('105'): Max NLM = 379.9
Line 106 ('106'): Max NLM = 980.3
Line 107 ('107'): Max NLM = 456.7
Line 108 ('108'): Max NLM = 856.8
Line 109 ('109'): Max NLM = 445.9
Line 110 ('110'): Max NLM = 728.6
Line 111 ('111'): Max NLM = 176.8
Line 112 ('112'): Max NLM = 748.4
Line 113 ('113'): Max NLM = 268.1
Line 114 ('114'): Max NLM = 675
Line 115 ('115'): Max NLM = 460
Line 116 ('116'): Max NLM = 502.4
Line 117 ('117'): Max NLM = 346
Line 118 ('118'): Max NLM = 206.1
Line 119 ('119'): Max NLM = 335.8
Line 120 ('120'): Max NLM = 317.3
Line 121 ('121'): Max NLM = 1775
Line 122 ('122'): Max NLM = 460.8
Line 123 ('123'): Max NLM = 1688
Line 124 ('124'): Max NLM = 371.7
Line 125 ('125'): Max NLM = 411.4
Line 126 ('126'): Max NLM = 1672
Line 127 ('127'): Max NLM = 465.9
Line 128 ('128'): Max NLM = 1599
Line 129 ('129'): Max NLM = 1535
Line 130 ('130'): Max NLM = 379.6
Line 131 ('131'): Max NLM = 203.1
Line 132 ('132'): Max NLM = 600
Line 133 ('133'): Max NLM = 490.2
Line 134 ('134'): Max NLM = 364.2
Line 135 ('135'): Max NLM = 352.3
Line 136 ('136'): Max NLM = 142.2
Line 137 ('137'): Max NLM = 92.52
Line 138 ('138'): Max NLM = 197
Line 139 ('139'): Max NLM = 136.4
Line 140 ('140'): Max NLM = 4563
Line 141 ('141'): Max NLM = 2878
Line 142 ('142'): Max NLM = 2512
Line 143 ('143'): Max NLM = 95.66
Line 144 ('144'): Max NLM = 57.37
Line 145 ('145'): Max NLM = 65.43
Line 146 ('146'): Max NLM = 55.59
Line 147 ('147'): Max NLM = 98.67
Line 148 ('148'): Max NLM = 55.28
Line 149 ('149'): Max NLM = 139.3
Line 150 ('150'): Max NLM = 46.28
Line 151 ('151'): Max NLM = 42.52
Line 152 ('152'): Max NLM = 92.08
Line 153 ('153'): Max NLM = 23.54
Line 154 ('154'): Max NLM = 4456
Line 155 ('155'): Max NLM = 4976
Line 156 ('156'): Max NLM = 106.1
Line 157 ('157'): Max NLM = 105.3
Line 158 ('158'): Max NLM = 113.1
Line 159 ('159'): Max NLM = 94.87
Line 160 ('160'): Max NLM = 131.7
Line 161 ('161'): Max NLM = 36.76
Line 162 ('162'): Max NLM = 49.09
Line 163 ('163'): Max NLM = 69.66
Line 164 ('164'): Max NLM = 202
Line 165 ('165'): Max NLM = 260
Line 166 ('166'): Max NLM = 291.8
Line 167 ('167'): Max NLM = 207.8
Line 168 ('168'): Max NLM = 208
Line 169 ('169'): Max NLM = 741
Line 170 ('170'): Max NLM = 785.8
Line 171 ('171'): Max NLM = 1333
Line 172 ('172'): Max NLM = 1442
Line 173 ('173'): Max NLM = 1287
Line 174 ('174'): Max NLM = 932.3
Line 175 ('175'): Max NLM = 792.3
Line 176 ('176'): Max NLM = 13.69
Line 177 ('177'): Max NLM = 27
Line 178 ('178'): Max NLM = 329.6
Line 179 ('179'): Max NLM = 574
Line 180 ('180'): Max NLM = 640.6
Line 181 ('181'): Max NLM = 398
Line 182 ('182'): Max NLM = 380.6
Line 183 ('183'): Max NLM = 41.6
Line 184 ('184'): Max NLM = 505.3
Line 185 ('185'): Max NLM = 592.2
Line 186 ('186'): Max NLM = 1563
Line 187 ('187'): Max NLM = 1343
Line 188 ('188'): Max NLM = 958.1
Line 189 ('189'): Max NLM = 369.1
Line 190 ('190'): Max NLM = 497
Line 191 ('191'): Max NLM = 500
Line 192 ('192'): Max NLM = 457.8
Line 193 ('193'): Max NLM = 201.7
Line 194 ('194'): Max NLM = 116.6
Line 195 ('195'): Max NLM = 102.4
Line 196 ('196'): Max NLM = 683.3
Line 197 ('197'): Max NLM = 554.2
Line 198 ('198'): Max NLM = 143.5
Line 199 ('199'): Max NLM = 155.8
Line 200 ('200'): Max NLM = 156.5
Line 201 ('201'): Max NLM = 162.4
Line 202 ('202'): Max NLM = 160
Line 203 ('203'): Max NLM = 124.5
Line 204 ('204'): Max NLM = 506.6
Line 205 ('205'): Max NLM = 564.1
Line 206 ('206'): Max NLM = 1809
Line 207 ('207'): Max NLM = 1481
Line 208 ('208'): Max NLM = 636.9
Line 209 ('209'): Max NLM = 228.1
Line 210 ('210'): Max NLM = 233.5
Line 211 ('211'): Max NLM = 868.1
Line 212 ('212'): Max NLM = 783.2
Line 213 ('213'): Max NLM = 527.6
Line 214 ('214'): Max NLM = 148
Line 215 ('215'): Max NLM = 146.6
Line 216 ('216'): Max NLM = 1693
Line 217 ('217'): Max NLM = 1638
Line 218 ('218'): Max NLM = 700.6
Line 219 ('219'): Max NLM = 652
Line 220 ('220'): Max NLM = 646.3
Line 221 ('221'): Max NLM = 353.6
Line 222 ('222'): Max NLM = 311
Line 223 ('223'): Max NLM = 69.81
Line 224 ('224'): Max NLM = 47.36
Line 225 ('225'): Max NLM = 35.86
Line 226 ('226'): Max NLM = 96.23
Line 227 ('227'): Max NLM = 80.29
Line 228 ('228'): Max NLM = 325
Line 229 ('229'): Max NLM = 337.5
Line 230 ('230'): Max NLM = 392.7
Line 231 ('231'): Max NLM = 366.3
Line 232 ('232'): Max NLM = 288.2
Line 233 ('233'): Max NLM = 113
Line 234 ('234'): Max NLM = 84.12
Line 235 ('235'): Max NLM = 66.68
Line 236 ('236'): Max NLM = 92.89
Line 237 ('237'): Max NLM = 95.88
Line 238 ('238'): Max NLM = 298
Line 239 ('239'): Max NLM = 304.4
Line 240 ('240'): Max NLM = 351.2
Line 241 ('241'): Max NLM = 263.4
Line 242 ('242'): Max NLM = 298.6
Line 243 ('243'): Max NLM = 588.2
Line 244 ('244'): Max NLM = 611.4
Line 245 ('245'): Max NLM = 746.5
Line 246 ('246'): Max NLM = 1383
Line 247 ('247'): Max NLM = 1686
Line 248 ('248'): Max NLM = 1208
Line 249 ('249'): Max NLM = 1743
Line 250 ('250'): Max NLM = 1650
Line 251 ('251'): Max NLM = 82.07
Line 252 ('252'): Max NLM = 86.18
Line 253 ('253'): Max NLM = 418.6
Line 254 ('254'): Max NLM = 629.4
Line 255 ('255'): Max NLM = 706.7
Line 256 ('256'): Max NLM = 1763
Line 257 ('257'): Max NLM = 1479
Line 258 ('258'): Max NLM = 935.1
Line 259 ('259'): Max NLM = 435.8
Line 260 ('260'): Max NLM = 574.7
Line 261 ('261'): Max NLM = 246.2
Line 262 ('262'): Max NLM = 229.1
Line 263 ('263'): Max NLM = 60.6
Line 264 ('264'): Max NLM = 369.3
Line 265 ('265'): Max NLM = 422.1
Line 266 ('266'): Max NLM = 77.81
Line 267 ('267'): Max NLM = 80.71
Line 268 ('268'): Max NLM = 320.6
Line 269 ('269'): Max NLM = 547.9
Line 270 ('270'): Max NLM = 612.2
Line 271 ('271'): Max NLM = 353.8
Line 272 ('272'): Max NLM = 331.4
Line 273 ('273'): Max NLM = 9.943
Line 274 ('274'): Max NLM = 295.9
Line 275 ('275'): Max NLM = 339.9
Line 276 ('276'): Max NLM = 922.4
Line 277 ('277'): Max NLM = 811.9
Line 278 ('278'): Max NLM = 221.2
Line 279 ('279'): Max NLM = 82.39
Line 280 ('280'): Max NLM = 57.09
Line 281 ('281'): Max NLM = 1114
Line 282 ('282'): Max NLM = 971.9
Line 283 ('283'): Max NLM = 354.8
Line 284 ('284'): Max NLM = 64.83
Line 285 ('285'): Max NLM = 80.99
Line 286 ('286'): Max NLM = 99.65
Line 287 ('287'): Max NLM = 103.5
Line 288 ('288'): Max NLM = 170.3
Line 289 ('289'): Max NLM = 563.7
Line 290 ('290'): Max NLM = 612.7
Line 291 ('291'): Max NLM = 921.4
Line 292 ('292'): Max NLM = 979.5
Line 293 ('293'): Max NLM = 955.7
Line 294 ('294'): Max NLM = 465.1
Line 295 ('295'): Max NLM = 709.7
Line 296 ('296'): Max NLM = 1028
Line 297 ('297'): Max NLM = 816.3
Line 298 ('298'): Max NLM = 779.7
Line 299 ('299'): Max NLM = 302.8
Line 300 ('300'): Max NLM = 397.1
"""

# --- Data Parsing ---
lines_grouped_index = {}
for line_entry in raw_output_grouped_index.strip().split('\n'):
    match = re.match(r"Line (\d+) => Mahalanobis distance = ([\d\.]+)", line_entry)
    if match:
        line_num = int(match.group(1))
        maha_dist = float(match.group(2))
        lines_grouped_index[f"L{line_num}"] = maha_dist

lines_nlm = {}
for line_entry in raw_output_nlm.strip().split('\n'):
    if not line_entry.startswith("Line"):
        continue
    match = re.match(r"Line (\d+).* Max NLM = ([\d\.e\+]+)", line_entry)
    if match:
        line_num = int(match.group(1))
        nlm_value = float(match.group(2))
        lines_nlm[f"L{line_num}"] = nlm_value

# --- Data Preparation for Plotting ---
sorted_gi_items = sorted(lines_grouped_index.items(), key=lambda item: item[1], reverse=True)
num_top_lines = 8
lines_to_plot_data = {}

for i in range(min(num_top_lines, len(sorted_gi_items))):
    lines_to_plot_data[sorted_gi_items[i][0]] = sorted_gi_items[i][1]

key_lines_to_ensure = ["L14", "L4"]
for kl in key_lines_to_ensure:
    if kl in lines_grouped_index:
        lines_to_plot_data[kl] = lines_grouped_index[kl]

final_plot_items = sorted(lines_to_plot_data.items(), key=lambda item: item[1], reverse=True)

line_keys = [item[0] for item in final_plot_items]
line_names_plot = [key.replace("L", "Line ") for key in line_keys]
grouped_index_plot = [item[1] for item in final_plot_items]
nlm_plot = [lines_nlm.get(key, 0) for key in line_keys]

# --- Plotting Enhancements ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(12, 5)) # Increased figure size for clarity

# Define x-axis positions and bar width
x = np.arange(len(line_names_plot))
width = 0.4

# Define contrast colors
color_gi = 'firebrick'
color_nlm = 'steelblue'

# Plot Grouped Index on the primary y-axis (ax1)
bars1 = ax1.bar(x - width/2, grouped_index_plot, width, label='Grouped Index', color=color_gi, zorder=3)
ax1.set_xlabel("Line", fontweight='bold')
ax1.set_ylabel("Grouped Index", color=color_gi, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_gi)
ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)

# Create a secondary y-axis for NLM
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, nlm_plot, width, label='Maximum NLM', color=color_nlm, zorder=3)
ax2.set_ylabel("Maximum NLM", color=color_nlm, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_nlm)
# --- APPLY NEW STYLE ---
# 1. Add a full, solid grid behind the bars
ax1.grid(True, which='major', linestyle='-', color='#b0b0b0', alpha=0.7, zorder=0)
# 2. Remove the top and right borders (spines)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
# --- ADD LABELS TO BARS ---
# Add labels for the Grouped Index bars
ax1.bar_label(bars1, fmt='{:,.0f}', padding=3, color=color_gi, fontsize=10, fontweight='bold')
# Add labels for the NLM bars
ax2.bar_label(bars2, fmt='{:,.0f}', padding=3, color=color_nlm, fontsize=10, fontweight='bold')

# --- ADJUST Y-LIMITS TO MAKE SPACE FOR LABELS ---
ax1.set_ylim(top=max(grouped_index_plot) * 1.18)
ax2.set_ylim(top=max(nlm_plot) * 1.18)

# Set x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(line_names_plot, rotation=45, ha="right", rotation_mode="anchor")

# Add a title and combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# --- MOVED LEGEND TO PREVENT OVERLAP ---
ax1.legend(handles=handles1 + handles2, loc='upper right', fontsize=18)

# --- Add Annotation ---
try:
    line4_index = line_names_plot.index("Line 4")
    ax2.annotate('NLM mistakenly identifies Line 4',
                 xy=(x[line4_index] + width/2, nlm_plot[line4_index]),
                 xytext=(x[line4_index] + 1.5, nlm_plot[line4_index] * 0.95), # Adjusted position
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=16,
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.9),
                 zorder=10)
except ValueError:
    print("Annotation for Line 4 skipped: Line not in plot.")

fig.tight_layout()
fig.savefig("GroupedIndex_vs_NLM_with_labels.pdf", bbox_inches="tight")
plt.show()