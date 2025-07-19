from matplotlib.image import imread


img_full_pitch = imread('assets/football_pitch.png')
img_half_pitch = imread('assets/football_pitch_half.png')

PASS_LINE_COLORS = {
    'vertical forward': 'green',
    'vertical backward': 'red',
    'diagonal forward': 'blue',
    'diagonal backward': 'orange',
    'horizontal': 'grey'
}

PITCH_LENGTH = 100
PITCH_WIDTH = 65
GOAL_WIDTH = 7.32
Y_LEFT_POST = 50 - (GOAL_WIDTH / (PITCH_WIDTH * 2) * 100)
Y_RIGHT_POST = 50 + (GOAL_WIDTH / (PITCH_WIDTH * 2) * 100)

XDRAW_TOLERANCE = 0.25
