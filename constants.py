"""
    Config for paths, joint set, and normalizing scales.
"""


class usernamel:
    night = ['night_random_10_up1', 'night_random_28_up2', 'night_random_34_up1']
    two = ['0106tworandom6_down2', '0106tworandom14_up1', '0106tworandom21_up2']
    ten = ['0105tenrandom34_0', '0105tenrandom26_0', '0105tenrandom25_down3']
    five = ['0105fiverandom17_up2', '0105fiverandom21_0', '0105fiverandom9_down2']
    seven = ['0106sevenrandom0_up1', '0106sevenrandom21_up2', '0106sevenrandom6_up1']
    six = ['0104sixrandom8_down3', '0104sixrandom23_down1', '0104sixrandom33_up2']
    four = ['0104fourrandom29_0', '0104fourrandom13_up3', '0104fourrandom7_down1']
    eleven = ['eleven_random_70_down1', 'eleven_random_13_up2', 'eleven_random_57_0']
    three = ['0106threerandom30_down1', '0106threerandom7_down2', '0106threerandom3_0']
    twelve = ['twelve_random_44_up1', 'twelve_random_31_up4', 'twelve_random_57_0']
    eight = ['eight_random_0_down1', 'eight_random_42_up1', 'eight_random_69_up1']


class motionnamel:
    clap = ['0106CZPclap-3-0', '0106CZPclap-11-up1', '0106CZPclap-15-up4']
    jump = ['0106CZPjump-25-down2', '0106CZPjump-3-0', '0106CZPjump-34-up2']
    run = ['0106CZPrun-11-up1', '0106CZPrun-13-down4', '0106CZPrun-15-up4']
    walk = ['0106CZPwalk-3-0', '0106CZPwalk-15-up4', '0106CZPwalk-34-up2']


file_dir = '../dispaddata/'
window = 1
seq = 30
ColumSuSmx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'circular', 'lateral', 'rawname']
ColumMuMmx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'username', 'circular',
              'lateral', 'motion', 'rawname', 'upperc', 'lowerc', 'height', 'weight', 'age']
ColumSuMmx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'circular', 'lateral', 'motion', 'rawname']
