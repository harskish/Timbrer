---------------------------------------------------------------------
���̃t�@�C���́ATiMidity++ �� Tcl/Tk �C���^�t�F�C�X TkMidity �ɂ���
�������Ă��܂��B

TkMidity �́A�p�l���\���ɂ�� TiMidity++ �{�̂ƑΘb�I�ȓ�����s������
�̃C���^�t�F�[�X�ł��BTk ���g�p���Ă��܂��̂ŁAMotif �̂悤�Ȕ�����
�\����ʂ��AMotif ���C�u���������ŗ��p���邱�Ƃ��ł��܂��B

* WHAT'S NEW in 1.5

- timer callback ���g�p�����g���[�X�\��
- Forward / Backward �{�^��
- ��������̃o�O�t�B�b�N�X..


* CONTENTS

���̃A�[�J�C�u�ɂ͈ȉ��̂��̂������Ă��܂�:

README.tk	- ���̃t�@�C��
tk_c.c		- tk-interface �̃R���g���[���̃\�[�X�R�[�h
tkmidity.ptcl	- ���C���p�l�� (preprocess �O)
tkpanel.tcl	- ���C���p�l��
browser.tcl	- �t�@�C���u���E�U
misc.tcl	- �e��R�[�h
tkbitmaps/*.xbm	- TkMidity �p�r�b�g�}�b�v�t�@�C��

(�� : ����̓I���W�i���̔z�z�ł̓��e�ł��BTiMidity++ �ł́A
�S�Ẵt�@�C�������炩���ߓK�؂ȃf�B���N�g���ɓ����Ă��܂�)


* USAGE

TkMidity �ɂ� 4 �̃��[�h (repeat , shuffle, auto-start, auto-exit)
������܂��B
"Repeat" �ł́A�w��t�@�C�����S�ĉ��t���ꂽ��A�ŏ��ɖ߂�Ăщ��t��
�J�n���܂��B
"Shuffle" �ł́A�w��t�@�C���̒����烉���_���ɉ��t����t�@�C����
�I��ŉ��t���܂��B
"Auto-start" �ł́ATkMidity ���N������Ɠ����ɉ��t���J�n���܂��B
"Auto-exit" �ł́A�S�Ă̋Ȃ̉��t���I�����玩���I�� TkMidity ���I��
���܂��B
������̐ݒ���A"Save Config" ���j���[�ŃZ�[�u���Ă������Ƃ��ł��܂��B

�f�B�X�v���[��̐ݒ�� "Display" ���j���[�ŕύX���邱�Ƃ��ł��܂��B
���̐ݒ���A"Save Config" ���j���[�ŃZ�[�u����A����N�����ɐݒ肪
�Č�����܂��B

ver.1.3 ����A�t�@�C���̃I�[�v��/�N���[�Y �̃��j���[�ƁA�L�[�{�[�h
�V���[�g�J�b�g���T�|�[�g����Ă��܂��B���t����ȖڂɁA�C�ӂ̃t�@�C����
�ǉ��ł���悤�ɂȂ��Ă��܂��B
�܂��A�L�[�{�[�h�V���[�g�J�b�g�͈ȉ��̂悤�ɂȂ��Ă��܂�:

[Enter]		: ���t�J�n
[Space]		: �ꎞ��~ / ���t�ĊJ
[c]		: ��~
[q]		: TkMidity �̏I��
[p] or [Left]	: �O�̋�
[n] or [Right]	: ���̋�
[v] or [Down]	: �{�����[������ (5%)
[V] or [Up]	: �{�����[���グ (5%)
[F10]		: ���j���[���[�h��
[Alt]+[Any]	: ���j���[�̓��e��I��

ver.1.4 ����A�g���[�X�\�����T�|�[�g����܂����B�e MIDI �`�����l������
�{�����[����p���|�b�g�̓��������A���^�C���Ɍ��邱�Ƃ��ł��܂��B����
�\��������Ƃ��́ATiMidity++ �̋N���I�v�V�����ɓK�؂ȃt���O���w�肵��
��������(�ڍׂ̓}�j���A�����Q��)�B
(��: `-ikt' �� t �����ɕt���܂�)


* PROGRAM NOTES

���� version �ł́ATcl7.5 ����� Tk4.1 �̃��C�u�������A���ꂼ��K�v�ł��B
�Â� version �ł� wish ���g�p���Ă��܂������A���݂� version �ł͒���
���C�u�����������N���Ă��܂��B
�܂��A���p����ۂɂ́Ashared memory �ւ̃A�N�Z�X�����K�v�ł��B


* TROUBLE SHOOTING

+���t����ہA���̃t�@�C�������݂��邱�Ƃ� TiMidity++ �ɓn���O��
�X�N���v�g���Ŋm�F���Ă��܂����A�H�Ɂu�t�@�C�������݂��܂���v�̂悤��
�G���[���o�邱�Ƃ�����܂��B

		Takashi Iwai	<iwai@dragon.mm.t.u-tokyo.ac.jp>
				<http://bahamut.mm.t.u-tokyo.ac.jp/~iwai/>

[�e�L�X�g�̖|��͒����� <breeze_geo@geocities.co.jp> ���s���܂���]
