import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches

# Raw Mahalanobis distance data
raw_output = """
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
Line 14 has the largest Mahalanobis distance: 22756.695655
Largest Lagrangian param => index=0, value=1.93e+04, line=4, error_type=R
"""

# Parse Mahalanobis distances
lines_maha_raw = {}
for line_entry in raw_output.strip().split('\n'):
    match = re.match(r"Line (\d+) => Mahalanobis distance = ([\d\.]+)", line_entry)
    if match:
        line_num = int(match.group(1))
        maha_dist = float(match.group(2))
        lines_maha_raw[f"L{line_num}"] = maha_dist

# --- Data Preparation for Plotting ---
sorted_maha_items = sorted(lines_maha_raw.items(), key=lambda item: item[1], reverse=True)
num_top_lines = 8
lines_to_plot_data = {}
for i in range(min(num_top_lines, len(sorted_maha_items))):
    lines_to_plot_data[sorted_maha_items[i][0]] = sorted_maha_items[i][1]

key_lines_to_ensure = ["L14", "L4"]
for kl in key_lines_to_ensure:
    if kl in lines_maha_raw:
        lines_to_plot_data[kl] = lines_maha_raw[kl]

context_lines_to_add = ["L1", "L150", "L300"]
for ctx_line in context_lines_to_add:
    if ctx_line in lines_maha_raw and ctx_line not in lines_to_plot_data:
        lines_to_plot_data[ctx_line] = lines_maha_raw[ctx_line]

final_plot_items = sorted(lines_to_plot_data.items(), key=lambda item: item[1], reverse=True)
line_names_plot = [item[0].replace("L", "Line ") for item in final_plot_items]
maha_distances_plot = [item[1] for item in final_plot_items]

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

fig, ax1 = plt.subplots(figsize=(12, 5))

colors = []
for name in line_names_plot:
    if name == "Line 14":
        colors.append('firebrick')
    elif name == "Line 4":
        colors.append('darkorange')
    else:
        colors.append('steelblue')

bars = ax1.bar(line_names_plot, maha_distances_plot, color=colors, width=0.6, zorder=3)
ax1.set_xlabel("Line")
ax1.set_ylabel("Grouped Index")
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

max_bar_height = max(maha_distances_plot) if maha_distances_plot else 1
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02 * max_bar_height,
                 f'{yval:,.0f}', ha='center', va='bottom', fontsize=16, weight='bold')

# --- Add Legend ---
perturbed_patch = mpatches.Patch(color='firebrick', label='Perturbed Line')
nlm_flagged_patch = mpatches.Patch(color='darkorange', label='NLM Flagged Line')
other_patch = mpatches.Patch(color='steelblue', label='Other Lines')
ax1.legend(handles=[perturbed_patch, nlm_flagged_patch, other_patch], fontsize=18)

plt.tight_layout()
fig.savefig("GroupedIndex_342bus.pdf",  bbox_inches="tight")       # vector preferred