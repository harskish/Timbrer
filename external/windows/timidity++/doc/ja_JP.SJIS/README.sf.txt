================================================================
** Timidity SoundFont Extension **

written by Takashi Iwai
<iwai@dragon.mm.t.u-tokyo.ac.jp>
<http://bahamut.mm.t.u-tokyo.ac.jp/~iwai/>

patch level 1: April 2, 1997
================================================================

* WHAT'S THIS?

TiMidity++ �� SoundFont ���̃T���v���f�[�^���g�p�ł���@�\��
�ǉ����邽�߂̊g���ł��B���t�p�̉��F�t�@�C���Ƃ��āA�I���W�i����
GUS patch �ɉ��� SoundFont ���g�p�ł���悤�ɂȂ�܂��B
SBK �� SF2 ������̃t�H�[�}�b�g���T�|�[�g���Ă��܂��B


* USAGE

�ݒ�t�@�C���ɁA�Q�̃R�}���h���V���ɒǉ�����Ă��܂��B

�g�p����T�E���h�t�H���g�t�@�C���́A�R���t�B�O�t�@�C����

	 soundfont sffile [order=number]

�Ə������ƂŐݒ�ł��܂��B�ŏ��̃p�����[�^ (sffile) �́A�g�p����
�t�@�C�����ł��B�t�@�C�����̂͑S�Ă̐ݒ��ǂݍ��񂾌�ɓǂݍ��܂�A
SoundFont �̓������ (sample data ������) �� TiMidity++ �̓�������
�ϊ�����܂��B

���̃p�����[�^�͏ȗ��\�ł����A���F�f�[�^��T�����Ԃ�ݒ肵�܂��B
`order=0' �̂Ƃ��́A�܂� SoundFont ��ǂݍ���ŁA���̌�� ����Ȃ�
�T���v���ɕt���Ă� GUS patch ����T���܂��B
`order=1' �̂Ƃ��́AGUS patch ��ǂ񂾌�� SoundFont ��ǂݍ��݂܂��B


`font' �R�}���h�́A�T���v���̌����ɂ��Ă̓����ݒ肵�܂��B
�����ASoundFont ���̂���T���v�����C�ɓ���Ȃ��A�g�������Ȃ��Ƃ��́A
���̃T���v���� `exclude' �T�u�R�}���h�Ŏw�肵�Ă��������B

	font exclude bank [preset [keynote]]

�ŏ��̃p�����[�^�́A�g�������Ȃ��T���v���� MIDI bank number �ł��B
���̎��͂��̃T���v���� MIDI program number �ł��B�h�����T���v����
���ẮA128 ���o���N�ԍ��Ɏw�肵����� drumset �� preset �ɁA
�C�ӂ̃h�����T���v���̃L�[�ԍ��� keynote �Ɏw�肵�Ă��������B
preset ����� keynote �͏ȗ��\�ł��B

�C�ӂ̃T���v�� (���邢�̓o���N) �ɂ��āA`order' �T�u�R�}���h�ɂ��
�������Ԃ�ς��邱�Ƃ��ł��܂��B

	font order number bank [preset [keynote]]

�ŏ��̃p�����[�^�͕ύX�������I�[�_�[�ԍ� (0 �܂��� 1) �ł��B����ȍ~��
���l��́A��L�� exclude �R�}���h�Ɠ��l�ł��B


* BUGS & TODO'S

- ����� bass drum �Ƀm�C�Y���ڂ�
- modulation envelope �̃T�|�[�g
- cut-off/resonance �̃T�|�[�g
- chorus / reverb �̃T�|�[�g


* CHANGES

- pl.1
	+ �{�����[���G���x���[�v�̌v�Z
	+ `font' �R�}���h�̒ǉ�
	+ font-exclude �R���g���[��

[�e�L�X�g�̖|��͒����� <breeze_geo@geocities.co.jp> ���s���܂���]
