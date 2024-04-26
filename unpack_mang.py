import lzma
import os
import struct
import sys

from dataclasses import dataclass

if len(sys.argv) != 2:
    print("usage: python unpack_mang.py [MangData_pak file]")
    sys.exit(1)

# XOR mask for "decryption"
XOR_MASK = [
    206645780, 723734864, 1239480316, 580225479, 792266171, 2003568016, 173935771, 407042348, 3970936712, 677952731, 2068556886, 3564191044,
    3091734357, 2895416559, 1983893172, 400227352, 3350655698, 1968120165, 2816813099, 3961211244, 3236935764, 1688953526, 1566345566, 2592177876,
    1930465221, 3948706648, 1941668660, 591721346, 2492473883, 3672887744, 1170734477, 3890439286, 3598367757, 2971225814, 3209415718, 3466839812,
    2026492460, 614984690, 2171430007, 3808407954, 2860329882, 3312818918, 3791624367, 1481649592, 32449893, 1471709714, 960930920, 2682706949,
    2567005570, 3642231969, 3605123788, 128087484, 118943478, 837378944, 2727715942, 2853406836, 3275407694, 516092737, 3970837620, 1487571174,
    3810369608, 2783775410, 2453964996, 3276716051, 1143347256, 1227114565, 4243675964, 1542769976, 1427537423, 1696028582, 163906188, 2836134858,
    2367037976, 1556349884, 3680307937, 3888472608, 3261994487, 1357825567, 1344479535, 2755769019, 2135152931, 1580168873, 3595341283, 484804801,
    4237663267, 1104534405, 3056146232, 3672695055, 1635009964, 2118712516, 934893318, 435222402, 633580622, 1259704600, 3027018502, 3653834904,
    1592375781, 2933050399, 3225764808, 986087690, 2333871370, 3607096572, 2163659081, 1674108445, 1624189344, 758889891, 813187562, 1078526983,
    341023315, 1193363706, 2933840959, 4116534461, 2693413634, 2365871405, 2229331166, 1700435390, 2415822173, 260064433, 2959007473, 4259794977,
    3404521582, 3448624093, 2037835463, 168791967, 3931763113, 269087992, 1117126494, 4266245806, 2147220943, 755536366, 3049934070, 3707632338,
    926842437, 3501263491, 3054331303, 1852608975, 1960567327, 3607094645, 2124781323, 911399376, 250650936, 600131280, 1792570568, 2304813517,
    2359062015, 827104328, 1334011003, 3499224456, 2557797354, 1891143584, 2759381173, 3186863425, 420681475, 1078539893, 3366411270, 580801361,
    179422345, 2543939049, 2656125531, 3320828072, 3033386064, 808345766, 1840777502, 3918989822, 3406952872, 1444265074, 2363436829, 4089285052,
    925984693, 17031741, 1794752889, 3393494434, 3000009812, 2301132006, 3950341242, 1431652764, 1026134428, 2624322186, 2597835883, 3134814491,
    1653893488, 2390515348, 4208452394, 2895009447, 287692811, 2724596174, 3639121975, 999199731, 3862260120, 3835466355, 390684103, 2674376165,
    664697338, 1151225697, 785593508, 2886238912, 3133857824, 3051883788, 4164692967, 1700330919, 4032795676, 1316960250, 3377574730, 3694322709,
    2773150590, 944410670, 2183841191, 2916758260, 2196615867, 4102656316, 68635474, 279514029, 3512937379, 1420062578, 576555171, 276114641,
    1662743962, 1006353148, 2141359335, 1091247679, 933377686, 3829952755, 2180839361, 2900195033, 493910433, 1232153355, 4241508026, 3424398936,
    1381879358, 3002769957, 1344632040, 485413058, 3368295059, 927283511, 4065580529, 2428834022, 3239686320, 4078125331, 3399122257, 3587969828,
    194763849, 1618661677, 3038699606, 438481936, 810317870, 1674391885, 2674139996, 1616647138, 1420822430, 341165402, 2389586095, 762598781,
    3632940506, 1628403144, 1764155742, 3712031937, 1038693730, 3630157000, 3881267717, 1446178400, 3462785086, 4251779385, 3507781919, 2274716825,
    3849550134, 1843225890, 3608482772, 3157812467, 2772326182, 1974809394, 1676700943, 1949926986, 809958246, 2029430421, 1125618809, 708816193,
    4168270042, 3173633418, 2162658834, 595341137, 270348546, 1081137260, 2898016442, 3284511452, 2066206726, 2140826182, 822024139, 3684850237,
    1717735787, 3391913565, 3214691887, 837042856, 1855183835, 3158469670, 492269501, 762217151, 1704530633, 546411167, 412208485, 1675471202,
    272461814, 1425007759, 2606718417, 3364062053, 2089153866, 3492472674, 2612470299, 2002575440, 1741853569, 2850965337, 2048334869, 2919997072,
    3999206068, 2014437926, 2643352521, 1965652633, 2017589980, 1822541231, 3982334714, 3970152386, 4066310243, 2448268024, 3169717433, 158582432,
    2760168408, 2942532784, 85986144, 3708772871, 2581992445, 2114172521, 3364660144, 1612501040, 4169534049, 2431002436, 1973125701, 3503286380,
    3887568528, 2359438436, 1569951358, 3982534225, 1563654051, 4225136761, 2410066449, 1167525327, 1906841931, 523788370, 2651139984, 2069874618,
    684549479, 3131484911, 4150979719, 2773502943, 3770016698, 91138476, 2251124618, 3892548734, 4039274386, 3279580037, 3238522923, 4098551619,
    3703480041, 175619336, 717650144, 337427621, 3989442359, 4241504107, 2949766670, 2086300014, 773335599, 355021866, 1756295962, 573329898,
    905968335, 1919820298, 1686982270, 17434622, 4220548465, 2791689907, 4225506137, 1942596152, 2946323162, 2277326267, 2665049209, 20493146,
    1273567118, 1706910096, 842090755, 331831699, 2959934375, 3646791103, 3655428131, 561214017, 284945904, 1254317548, 1170245033, 115352441,
    3531562645, 747917810, 3789439118, 2844840968, 2018497737, 2132832568, 614140532, 757024368, 1345625483, 3745365838, 2653636016, 1180303068,
    1414430214, 2850071232, 2777301494, 937772263, 2359707082, 3825107459, 734710535, 3631487571, 1171918375, 3963883336, 3340445215, 3186813543,
    3660420147, 1135117457, 232117781, 2250562691, 160826210, 3682305599, 1452605150, 2963939757, 970178324, 1844733601, 2478580009, 3793306275,
    3156483030, 3038898540, 1000765493, 4050476749, 3446123570, 371113942, 1089215497, 1049963665, 3732894115, 873889042, 3214428767, 1221738780,
    454230434, 2831992487, 758872381, 1485092901, 3673604513, 1196030985, 2693920862, 3271914985, 2834914160, 2545151828, 264554561, 3272993244,
    4266070019, 3322350334, 4185990901, 4010842178, 2891145218, 3141732792, 1426411412, 3270448592, 849315107, 620512929, 3846199273, 923393068,
    2709849251, 955053081, 4253291904, 2010082056, 169685184, 763096389, 333364990, 3354128884, 2034384320, 3450978702, 422806487, 3290437161,
    2797173039, 1615644419, 1583004729, 301708667, 3871960967, 159238304, 3567142543, 3940302494, 2638296612, 634407318, 1481844660, 3360394309,
    2590422118, 3797592326, 3716644980, 1784010140, 3185978529, 3117400217, 763693054, 3329222238, 3072465841, 3051408860, 1719908347, 808431593,
    675158296, 4182354988, 555456506, 3264660293, 2093936488, 2203441101, 1418038873, 3939259042, 1905531624, 3261369168, 115796636, 1195949659,
    3243031058, 1795401253, 1758217232, 1732452431, 264525218, 4154822802, 1682866386, 742843243, 2314724680, 3315700630, 4073651102, 625264605,
    1664579882, 1221163824, 4061250798, 1529854231, 2688088188, 1450534506, 4257445009, 680303842, 1646689933, 2048398450, 3021043897, 4253065330,
    1034952535, 1171240738, 271802892, 3143831662, 60735503, 2599306772, 2977906354, 2118884615, 2522506548, 18269746, 3002718823, 3970120066,
    3171188643, 4139524677, 4108492683, 2385893840, 3934272803, 1002664966, 1942575536, 2403128214, 2475351283, 596073386, 253266220, 2692096834,
    2059598565, 1734919930, 3802512038, 959528854, 2243715423, 1986382299, 2454407982, 3999098944, 2299812213, 1076094946, 417874496, 1125809248,
    3727097456, 3163939059, 2602509459, 3928206527, 492523172, 166126802, 3984044530, 681468028, 779663739, 1657234645, 3677502644, 2813566958,
    2818751136, 501691725, 1650113074, 198006708, 3809713407, 3753045959, 2753206856, 3021528688, 1198048203, 4116234401, 3284766823, 244337502,
    2591524372, 2707928345, 578741921, 3784685023, 3581442002, 1733462874, 3206139581, 2734069392, 4068337670, 1349666429, 2935440742, 3027404468,
    3883848621, 558973408, 2662342068, 1209370596, 3181687193, 873999873, 1498438529, 2501197563, 231104940, 2718697889, 1580751579, 91208588,
    1887189759, 3708388879, 2195206086, 1132826447, 880070299, 59030131, 2313843990, 2133113557, 2294931651, 2312297696, 2054742767, 827719106,
    3329040446, 2779747937, 3577340805, 508516524, 2287994254, 3742290875, 1124062735, 3347461555, 2009803523, 1747685256, 1590682196, 1027749642,
    2626623430, 1654401726, 2481407044, 3886812686, 2999717740, 2967582403, 3763444733, 1591591251, 3000045187, 3618115141, 1491364818, 3118824364,
    442205574, 3812036027, 3900909580, 1661514637, 2393439736, 1272362297, 2021277262, 2370413828, 214495113, 1435430234, 2340569746, 4086131486,
    460880117, 2332256869, 411015903, 1959444514, 3745444988, 1101801349, 767399460, 1285633222, 1592717693, 1092803143, 3146583621, 1899258550,
    1264308187, 768504603, 1051083183, 3062312978, 261939939, 1625528641, 170819223, 1211779762, 3824007227, 2875290266, 410830191, 230982379,
    1995965792, 774525238, 3773611403, 3762592605, 2776651984, 3759047823, 3825841878, 1241757971, 207294913, 295016931, 1408888889, 311028638,
    3624381366, 823079358, 3407149361, 162822696, 439719181, 673566729, 3169480800, 1581233666, 2354935180, 2657922081, 694147612, 1809272242,
    3194873642, 1432645531, 468681313, 1547944266, 1589770741, 2596534289, 1916137829, 991478686, 3903305443, 1109161267, 1231745823, 1667201611,
    3732830166, 3607849879, 622362685, 1359978960, 3204294190, 2057757776, 1973897009, 2132944122, 1480658001, 823115521, 1600912448, 849470568,
    534355654, 897317097, 2469833873, 3409514784, 4066268093, 1154482558, 2540144705, 2563109025, 1875575930, 1433369563, 2125297274, 154048009,
    1345574950, 1858357541, 3758032743, 3615646508, 3736778361, 1100628483, 2080220464, 2570407464, 3087917194, 2823455534, 1440485911, 2027425412,
    1962601248, 1856539628, 2600122605, 602191346, 1417562533, 3932675126, 3315084106, 17255473, 2783984735, 1691098424, 128541611, 628927009,
    723308618, 1216160833, 1759530391, 2928026173, 631382878, 438095904, 3629091293, 929508938, 654912927, 3234909871, 32653082, 670318946,
    1665800362, 1474228077, 2462673872, 2174983813, 4234578250, 4231296258, 374944757, 3029933613, 1319982318, 2103523532, 2471843874, 3117345977,
    3824949104, 1688040116, 793274777, 1070379995, 3539696774, 1876370061, 1667759504, 3284447002, 4279721876, 503700668, 1013295703, 3634908521,
    2421023802, 1222015335, 1779727551, 3391775425, 3108131425, 1707401471, 2607695536, 1951170228, 2438126543, 1713462873, 1152478771, 242498811,
    1562079542, 2082893879, 1852097539, 2004324357, 1458828804, 2792580443, 1036692995, 1849852633, 3836572457, 3935065775, 2960145923, 3864153122,
    2063925091, 1464320409, 1658574886, 2967962182, 2628410313, 1012257313, 3714702030, 2374830393, 3421900322, 2623946643, 2382774088, 3211792107,
    2108604464, 614626728, 4073053045, 910596905, 2471373447, 3672763642, 366886258, 1190301828, 2531126386, 3565123078, 75834198, 1981182565,
    1687164096, 1846023769, 267144864, 288103178, 990880167, 1745232528, 3533411805, 133952278, 442738363, 1198917837, 1866628210, 2641428002,
    677555580, 4032343727, 4047342343, 205732759, 2403188023, 3089327302, 3041362938, 3327193725, 1437640999, 2493340102, 1470225547, 89406805,
    1651568430, 2857212976, 571310744, 2583054172, 507137508, 3216861020, 2719215014, 1505369977, 2992852382, 3962713058, 2438303420, 3638968729,
    1499699064, 1899466028, 2661686258, 3004391363, 2875802733, 2134107902, 1100242952, 3853859490, 4083624032, 2364825361, 184226028, 2512087194,
    4196903091, 2747538456, 4028077010, 1695461079, 1574176336, 3027538392, 629824066, 1160333658, 1419096383, 3790494134, 3799689381, 1068187662,
    202447536, 3544776261, 924380250, 1289926097, 3518869654, 4227708750, 713889866, 1991575640, 2095944326, 2434680135, 2802361894, 1363825510,
    474400592, 946616930, 2629470906, 2589861750, 458845375, 4144525547, 478126146, 3960256225, 2848813918, 2924022255, 1615806069, 497754636,
    2188470140, 1590572106, 2647587362, 266977516, 1145456070, 4254303414, 1356892475, 4266816676, 1032996184, 3329966431, 2352447459, 3647207692,
    1845885458, 97307861, 1588659776, 534688654, 2001973532, 2526389260, 172203084, 4169270083, 51774817, 487671271, 839250069, 4111495703, 497515745,
    1872208984, 1769743973, 3711577447, 3153276740, 157397483, 3437013508, 2224765250, 3714853545, 4138430102, 3360513494, 856363964, 1634896195,
    2010658358, 989807098, 3026498265, 2005115126, 4111611602, 947561856, 3305790956, 229609317, 867688369, 4267561084, 1562330579, 2746029562,
    309703786, 1165676439, 2140754304, 434856064, 3964542319, 2271003421, 231700621, 2115623463, 1588539046, 1549776394, 327747917, 3820781341,
    377488370, 3266798882, 428899160, 2848138525, 749700935, 1905039096, 1967047582, 1351472122, 3888466099, 637396231, 2559385335, 1224904245,
    410437882, 4131099830, 3769426868, 1142663433, 3702311582, 638703993, 1745540392, 2247824507, 2082153324, 3908515277, 1432580160, 2876714862,
    3000109331, 1101576540, 693713427, 2326566830, 2101896753, 3290817530, 2482934891, 4161084075, 4097295798, 3197759884, 1253969994, 3519855967,
    3864196690, 1962591335, 2576996335, 652053747, 2595308195, 1727431962, 3695342455, 222352722, 4006994561, 2923176534, 3834959527, 2538403675,
    3644561092, 3461758097, 4260273549, 2889985787, 3009572180, 2185100436, 3225345933, 1339483325, 2330532532, 2321966092, 3809738482, 4114940500,
    1847050395, 2236946455, 2267903742, 4094327172, 1861068516, 1968169669, 1055230299, 2346999362, 2109442138, 209022508, 565387086, 310935255,
    3578096861, 1197741185, 1353459093, 977768670, 2798101608, 1123223432, 1237844985, 1432173998, 3328886910, 2989208408, 2316348285, 848456838,
    2611784223, 3501431174, 1460486223, 839930483, 1845379745, 3807986425, 3017465340, 767848434, 4251721240, 72780735, 312281414, 1987466627,
    595152019, 972885031, 1660744113, 3083661356, 2578722740, 1058299218, 1580286110, 3035160334, 3460615099, 1513550153, 3725813712, 644758393,
    3078068247, 2811453828, 648967814, 2190863220, 2597933044, 3983915263, 3050202371, 1363212630, 3829152212, 1750355020, 2534597794, 4194228557,
    1993016721, 2682390685, 922039141, 372162163, 1931252471, 790968920, 627828104, 2854387807, 1016418975, 1462573987, 1397716381, 1497776175,
    210935028, 2777675611, 1702493993, 2828893667, 2116617076, 2246341029, 1877356274, 4050460085, 2599827619, 425445571, 667887406, 315233617,
    4162220423, 1761146551, 1106231126, 834057460, 463909430, 867629784, 3146990525, 446889461, 2976163553, 1107766114, 841533980, 2290609877,
    1983112429, 3077243777, 759957284, 3162139871, 2512175571, 519597096, 1969339914, 1442349235, 2026195071, 741058825, 81557571, 3912419883,
    1860599156, 2659636826, 3405682931, 4088633005, 3422201692, 366967665, 2938027602, 2004079568, 4096871932, 1543730022, 3999274254, 1939315185,
    3686976641, 4036376149, 3382037159, 3856504731, 1370109515, 28416054, 1736279911, 1827825179, 3138391195, 757682081, 589137532, 2715364027,
    2568443312, 28263926, 3489246682, 4093175890, 3379194201, 1046965477, 262691791, 4021179856, 1682882820, 795577219, 2657204621, 2421955921,
    2167491917, 2028507350, 2827815999, 343654700, 2386311186, 3819396842, 332759062, 3374801026, 2648524412, 916201948, 1320648413, 3694199506,
    3068238233, 4210726895, 2861398404, 741568774, 561612386, 1438771221, 556121652, 715891537, 1405751261, 1498379345, 3347148793, 132410434,
    477495912, 3713159699, 4264678927, 3070891757, 2044699872, 3690105075, 1122418445, 1562286680, 381919611, 2052632512, 2906261909, 410453118,
    3025772650, 1191909388, 3678747538, 3637590634, 4175404657, 2290062651, 2972834370, 2950452287, 3157211506, 229102485, 810201934, 500131534,
    3704292093, 4228031701, 1141649363, 379029118, 3193104080, 3536843015, 709870425, 1211234221, 1369531628, 929961935, 2288843451, 2454104221,
    3484862814, 1332088926, 469442439, 883129475, 3037288699, 1099611282, 2905703779, 4024813368, 3088380817, 3206165497, 1800812479, 1393738377,
    3560355928, 1527116825, 3137913900, 2051905671, 3323671248, 155811744, 1296684615, 1780433078, 2966951423, 1958316416, 1361770401, 3886482005,
    2268944554, 1917140650, 3053264471, 3350020718, 1684211300, 3423310245, 4083579416, 1118426342, 3594290975, 2548304538, 1684621500, 995870992,
    2980215742, 4294612837, 150674131, 3824669219, 225397349, 657084477, 422911737, 1405328018, 2686384327, 1399547034, 1674907005, 551202509,
    3812152355, 2420770150, 3059651138, 690535028, 1677875046, 2258419142, 3509559141, 1382775707, 1170111615, 3339964199, 1848430280, 3444247527,
    229970665, 2605449671, 2487891239, 3982733335, 4132079682, 3529699274, 3885871778, 1993585428, 1133167315, 369221572, 2012037586, 1792026884,
    112813480, 913050447, 256132440, 1538704213, 1969694618, 4263047317, 2281069545, 1083483478, 3220873825, 2150999911, 2487463014, 977382459,
    1422271143, 1698934087, 391429348, 408404535, 1525252387, 424828582, 1570025271, 286768304, 2641530045, 2958985679, 2965128211, 2081088753,
    2024878494, 2584648505, 1763629052, 572876182, 1883697924, 73317034, 1536297731, 124794489, 1535909740, 1139311985, 364242103, 3256643390,
    3081014566, 2767544685, 880191175, 3473676002, 4086103517, 3188409889, 2946604676, 3555290324, 1168596271, 3220674116, 762900855, 4130555896,
    4099252022, 3161319503, 3451581042, 3008156480, 779978781, 2248792336, 2101111581, 537883064, 1411415469, 1910595253, 3354292488, 3336043288,
    3404191713, 799051493, 2683778617, 518504263, 3528269816, 2530077415, 932599809, 2750947472, 277241847, 3662958074, 2211482495, 4253395554,
    1388099283, 953247176, 2721296378, 2027416717, 1638157887, 3700931918, 2888795767, 1338505376, 2173582007, 2693378307, 2928596458, 1455362829,
    1343425798, 977496195, 1418623636, 1252420683, 281027484, 4195714089, 155840451, 2475784925, 3607608400, 2440812420, 2496195960, 422737669,
    4166486782, 2546052917, 3631854572, 3944267733, 4279535275, 2820947943, 3821026281, 1913730614, 4248022456, 1094144028, 382328758, 3117292309,
    535853841, 2482495271, 2848880077, 550258207, 1479152810, 2752829580, 1970354937, 249966703, 4144367833, 1642200725, 1576875171, 1071518770,
    4081552108, 718933539, 557244986, 3727857926, 1164324468, 3030665758, 686657595, 2409135510, 1598533604, 4261274085, 2496150193, 1390119215,
    3790805932, 3482194628, 3615729530, 3183106986, 999452868, 682442710, 1395343081, 3231403076, 3349550838, 3114901725, 2430426102, 1235056488,
    3790964454, 2482660914, 350621650, 3432020652, 3622748159, 4076649267, 1282477660, 4020968855, 817189029, 3737006820, 2826648724, 3804327033,
    2832723547, 238711780, 3580496109, 1228719150, 1949004813, 581870179, 389678816, 381552936, 2527478750, 878360898, 988790846, 2243497997,
    3006849507, 1313058488, 1750226188, 1732823521, 1470569796, 613183734, 2120428222, 3598287522, 291231139, 524803606, 4222192160, 1264408426,
    533598354, 3886733357, 2427541310, 3782152263, 996002649, 1218812993, 2770258492, 2847330647, 3545803609, 1980173406, 926801291, 1817807684,
    2532845894, 4164746478, 2434366062, 921010942, 3776996810, 2907092169, 4032149675, 2938573384, 3728074650, 2576430040, 1417640716, 2894827352,
    3717604306, 4095298561, 3660565110, 4123857015, 2840269235, 1362886548, 2149347617, 678812492, 2760470869, 844928386, 3323601802, 4280562648,
    2629968912, 926908785, 2918586492, 1888834696, 848081569, 2575949270, 57069921, 1075486884, 2153171915, 586238252, 1639685597, 2373814510,
    2446633391, 2044672597, 1355685959, 2534864592, 1246389640, 2623751708, 1794247650, 319315666, 785800013, 3326867795, 2711777684, 894417270,
    1351969013, 2579535503, 1641812362, 21751112, 732609927, 4015974459, 3836616648, 2614062704, 3922800041, 2889485728, 2160598842, 439999172,
    3203529266, 2339719681, 1971989571, 1375044179, 2963975859, 1834015261, 1108822859, 2786908661, 1759764625, 2964964174, 1902198485, 3765650275,
    98536720, 4158364297, 1757269642, 1972232890, 1749705440, 2133745789, 988913993, 3545468772, 2203727210, 2676576266, 2902570539, 524298564,
    4089942907, 3273444109, 3064248565, 103739760, 336959272, 861608409, 866852405, 3205872802, 687161948, 770091896, 1495702592, 1506600217,
    163909300, 1795573786, 1992656568, 1417599910, 3254948812, 2535919265, 4136650198, 3445489956, 616896266, 3939484346, 3008065931, 456199937,
    1115663993, 558702590, 295877040, 80079377, 161059717, 3619662000, 3651409337, 1550925308, 1933008924, 282082977, 2957697689, 1891705539,
    1512120506, 1065676617, 2027498687, 1499924076, 2974759211, 2714920861, 910733186, 4028867131, 1131829492, 567100306, 2882246846, 551375420,
    2098412143, 465103669, 2817182187, 2964872282, 1123712465, 3105053487, 3557798564, 496191083, 3102602051, 1000732131, 143760317, 1596531261,
    2586047208, 1226691445, 382597073, 3222693293, 2287836789, 1116476762, 2628760402, 779038129, 3572940863, 311078511, 3636824091, 2160728171,
    3597963975, 4130880382, 4014818938, 3554023095, 965808978, 3687894363, 1221012041, 1407692478, 1339510511, 2730638952, 3489999479, 2332318981,
    1520810129, 1949596460, 3568780827, 3086604535, 3812848408, 329079266, 1350948866, 2498389890, 2314512097, 676455305, 3021137855, 2917084186,
    3380890077, 2551163504, 1150546242, 2950003407, 3696850073, 1318426316, 1566687452, 3108971229, 4210996819, 2335093828, 3355785562, 1300009665,
    497035378, 1628434831, 1078108103, 1270869157, 2202130388, 4253588099, 2982500140, 568944000, 3193582796, 3721708676, 2440990794, 2909174824,
    1761844686, 3345608209, 1228882148, 2783101501, 2982075654, 3141765380, 214378380, 1674385581, 3014118271, 1122590152, 3116733404, 3204664911,
    639943528, 3438416882, 3229225562, 968380202, 4102666714, 503443607, 3142880846, 288817709, 1955411091, 2824376367, 43862090, 4177782636,
    1955867423, 1717691256, 1853702875, 4171729181, 1871386583, 780188708, 2695287204, 1279525859, 3008413711, 3321844773, 1406105318, 3217371369,
    3432124409, 247593856, 2988197956, 3229284444, 1538287185, 4116408857, 4127288887, 1591994449, 4285312736, 42865379, 2123415100, 2576141702,
    1806336567, 1993219883, 2739555085, 3050306907, 582512924, 1569167273, 1119017131, 1168573232, 2099496413, 1521122743, 1880694225, 2853914404,
    3263921112, 2868596792, 2129538850, 2281897897, 3943842748, 2518320006, 4224751452, 4099441716, 3746495712, 107868408, 1161088300, 1124779420,
    2201882106, 430501421, 2279570805, 2002713119, 847200960, 2654754315, 2928006953, 1930693062, 1270718366, 1651641285, 1812592385, 3062202411,
    2717041731, 1173460384, 1377351820, 1918406989, 1587783596, 2738128392, 65843800, 2963462354, 1506247558, 2756275613, 747848992, 2551148639,
    3549932918, 354754930, 574364694, 3682739925, 2212358462, 1412458391, 689201595, 3777242589, 848197784, 3716778956, 3751826098, 472467147,
    584516616, 835628754, 1661506146, 3414526837, 3844864083, 2560486092, 899073281, 2781530533, 2998551109, 113713218, 1941033486, 2030321315,
    3630436214, 3841843432, 1821439805, 4284169504, 2155297325, 643019283, 48934984, 2699333462, 1303537862, 409198382, 683261021, 2380119378,
    3816930655, 1725393945, 4023995636, 2280802005, 3275741941, 110977461, 833496846, 1542028399, 551621069, 2817698066, 3688859481, 220238737,
    1835278055, 371183916, 2449809799, 38457153, 932283361, 2942870691, 2403124400, 3688445604, 1498744429, 1514866421, 4101843340, 1176003743,
    1142640564, 1320066757, 4025210851, 2286764349, 1281813491, 1501982443, 3303479264, 1639495813, 1201485765, 2534845147, 336560816, 2294861228,
    1029895933, 1128403759, 912068975, 2258983099, 3600037949, 2385717007, 2851749314, 1703013721, 1271127163, 3397605804, 3574784206, 3919133827,
    3669422055, 832613631, 886233075, 2239768451, 3138666330, 3308918512, 1302136165, 293306345, 1923014324, 2349645326, 3076445029, 3070706191,
    60835856, 1432313882, 2295731864, 786772262, 3841410990, 833540475, 2647457406, 3183453772, 1396911960, 2962833603, 2967901331, 217250578,
    2035313567, 1109961262, 1480517943, 2255019881, 3178448982, 952777943, 1315135370, 513672305, 2601697818, 4020480855, 3119547504, 1303516640,
    3867531893, 273513218, 2671534473, 2620820089, 3718175472, 827757206, 1851665154, 922305992, 3367723418, 2832731915, 2167494005, 1444794879,
    3719876623, 2324731491, 94585526, 942926158, 3120047659, 2864030339, 2107807609, 1199878823, 602031979, 1420942734, 350351544, 3475797066,
    3020909057, 1472867728, 1372862741, 2322133907, 3312831677, 4018963547, 1858878596, 2475385454, 413830263, 4216802837, 3592353899, 824593733,
    1939175104, 148593768, 963685896, 4030803568, 1398509188, 1737196731, 3213023731, 2802579746, 1231332053, 859377382, 3638307549, 3309441645,
    2726897379, 2673422725, 3374476808, 4009558645, 2428550846, 333790396, 561688561, 4063327534, 1899545481, 3841488973, 745405485, 2711790982,
    3004501608, 3515508149, 2969866131, 477997813, 789105423, 1383991243, 1238932052, 2724383251, 1809336909, 1664465352, 589699974, 3849865757,
    4143598117, 1562508631, 1557801657, 3770125509, 3348454444, 3618755764, 2450727308, 1607159203, 3261840304, 3423198799, 611394825, 1798604676,
    3952846762, 3276410473, 2207306629, 1932479246, 3977154270, 2436559316, 764620569
]

C = ""
with open(sys.argv[1], "rb") as f:
    C = f.read()

off = 0
def unpack(fmt: str, data: bytes, size: int):
    global off
    out = struct.unpack(fmt, data[off:off+size])
    off += size
    if len(out) == 1:
        return out[0]
    return out

n_catalogs = unpack("=Q", C, 8)

def decrypt1(word, key) -> int:
    key = key & 0xff  # byte
    out = word >> (key & 0x1f) | word << 0x20 - (key & 0x1f)
    return out & 0xffffffff  # word

def decrypt2(word, key) -> int:
    word = word & 0xffffffff
    return word ^ XOR_MASK[key % 2048]

print("blob1 size:", n_catalogs*268)

decrypted_blob1 = bytes()
for key in range(n_catalogs * 268 // 4):
    x = unpack("=I", C, 4)
    x1 = decrypt1(x, key)
    x2 = decrypt2(x1, key)
    u = struct.pack("=I", x2)
    decrypted_blob1 += u

@dataclass
class MangDataCatalog:
    file_name: any
    key1: int
    key2: int
    offset: int
    uncompressed_size: int
    size: int

    def __post_init__(self):
        self.file_name = self.file_name.split(b"\x00")[0].decode("utf-8")

catalogs = []
for i in range(n_catalogs):
    unpacked = struct.unpack("=253s B H I I I", decrypted_blob1[i*268:(i+1)*268])
    catalog = MangDataCatalog(*unpacked)
    catalogs.append(catalog)

# https://stackoverflow.com/a/37400585
def decompress_lzma(data):
    results = []
    while True:
        decomp = lzma.LZMADecompressor(lzma.FORMAT_AUTO, None, None)
        try:
            res = decomp.decompress(data)
        except lzma.LZMAError:
            if results:
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it.
            else:
                raise  # Error on the first iteration; bail out.
        results.append(res)
        data = decomp.unused_data
        if not data:
            break
        if not decomp.eof:
            raise lzma.LZMAError("Compressed data ended before the end-of-stream marker was reached")
    return b"".join(results)

for catalog in catalogs:
    blob = C[catalog.offset:catalog.offset+catalog.size]
    decrypted = bytes()
    for i in range(catalog.size // 4):
        x = struct.unpack("=I", blob[i*4:(i+1)*4])[0]
        x1 = decrypt1(x, catalog.key1 + i)
        x2 = decrypt2(x1, catalog.key2 + i)
        u = struct.pack("=I", x2)
        decrypted += u
    decompressed = decompress_lzma(decrypted[1:])
    raw_path = catalog.file_name.replace("..\\", "")
    split_path = raw_path.split("\\")
    os.makedirs(os.path.join(*split_path[:-1]), exist_ok=True)
    file_path = os.path.join(*split_path)
    print("extracting", file_path)
    with open(file_path, "wb") as f:
        f.write(decompressed)


