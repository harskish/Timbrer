=======================================================================
                    ALSA�V�[�P���T�C���^�t�F�[�X
            Copyright (c) 2000  ��� �� <tiwai@suse.de>
=======================================================================

����
====

���̕����́CAdvanced Linux Sound Architecture(ALSA)�V�[�P���T�C���^�t�F
�[�X�Ɋւ�����̂ł��BALSA�V�[�P���T�C���^�t�F�[�X�́CALSA�V�[�P���T��
�R�A��timidity�ԂŒʐM���s���܂��B�C���^�t�F�[�X�̓V�[�P���T����C�x��
�g���󂯎��C(�ق�)���A���E�^�C���ŉ��t���܂��B
�{���[�h�ɂ����āCTiMidity��ALSA��̃\�t�g�E�F�AMIDI�V���Z�T�C�U�G���W
���Ƃ��āC�����Ƀ\�t�g�E�F�A���A���^�C��MIDI�����_�Ƃ��ē��삵�܂��B
���ׂẴX�P�W���[�����O�́CALSA�V�[�P���T�̃R�A�ɂ���čs����̂ŁC
�{�C���^�t�F�[�X�̓X�P�W���[�����O���[�`��������܂���B

ALSA�V�[�P���T�C���^�t�F�[�X���N������ɂ́C�ȉ��̂悤��timidity���N��
���Ă�������:
	% timidity -iA -B2,8 -Os -q0/0 -k0
�t���O�����g�T�C�Y�͒����\�ł��B��菭�Ȃ����قǃ��A���^�C�����X�|��
�X���ǂ��Ȃ�܂��B�����āCtimidity�͐V�����쐬���ꂽ�V�|�[�g�ԍ�(���L��
128:0�����128:1)��\�����܂��B
       ---------------------------------------
	% timidity -iA -B2,8 -Os -q0/0 -k0
	TiMidity starting in ALSA server mode
	Opening sequencer port 128:0 128:1
       ---------------------------------------
�����̃|�[�g�͑��̃V�[�P���T�|�[�g�Ɛڑ�����ł��܂��B�Ⴆ�΁Cpmidi��
�o�R����MIDI�t�@�C�������t����(�Ȃ�ĉߏ��:-)�ɂ́C
	% pmidi -p128:0 foo.mid
MIDI�t�@�C����2�̃|�[�g��K�v�Ƃ���ꍇ�́C���̂悤�ɐڑ����܂�:
	% pmidi -p128:0,128:1 bar.mid
�O����MIDI�L�[�{�[�h����ڑ�����ɂ́C����ȋ�ɂȂ�܂�:
	% aconnect 64:0 128:0

�C���X�g�[��
============

--enable-alsaseq��--enable-audio=alsa�I�v�V����������configure���Ă�
�������B�������C���̃I�[�f�B�I�f�o�C�X��C���^�t�F�[�X���ǉ����đI��
�܂��B

���ǂ����A���^�C�����X�|���X�𓾂�ɂ́Ctimidity��root�Ƃ��Ď��s����
���ł�(�ȉ����Q��)�Bset-UID root�́C�������������ł��ȒP�ȕ��@�ł��B
�C���X�g�[���ς�timidity�̃o�C�i���̃I�[�i�ƃp�[�~�b�V�������C�ȉ��̂�
���ɕύX����Ηǂ��ł��傤:
	# chown root /usr/local/bin/timidity
	# chmod 4755 /usr/local/bin/timidity

����ɂ���āC�Z�L�����e�B�E�z�[���������N���������m��Ȃ����ƂɋC����
���Ă�������!


���A���^�C�����X�|���X
======================

�C���^�t�F�[�X�́C�v���Z�X�X�P�W���[�����O��SCHED_FIFO�ɂ��āC�ł��邾
�������D��x�Ƀ��Z�b�g���邱�Ƃ����݂܂��BSCHED_FIFO���ꂽ�v���O�����́C
���ǂ����A���^�C�����X�|���X��悵�܂��B�Ⴆ�΁CSCHED_FIFO�������Ȃ�
timidity�́C/proc���A�N�Z�X����邽�тɒ������r�؂�������N���������m��
�܂���B
���̋@�\��L���ɂ���ɂ́Ctimidity��root�ŋN�����邩�Cset-uid root�ŃC
���X�g�[�����ׂ��ł��B


�C���X�c�������g�̃��[�h
========================

timidity�́C�v���O�����`�F���W�̃C�x���g����M���邽�тɁC�C���X�c����
���g�𓮓I�Ƀ��[�h���܂��B�Ƃ��ɂ���́C�Đ����̃o�b�t�@�A���_�[������
����āC�r�؂�������N�����܂��B����ɁC���ׂĂ̗\�񂪐ؒf�����ƁC
timidity�̓��[�h�����C���X�c�������g�����Z�b�g���܂��B���������āC�Đ�
�I��������[�h�����C���X�c�������g�����ׂăL�[�v����ɂ́Caconnect���o
�R����timidity�|�[�g�Ƀ_�~�[�|�[�g(�Ⴆ��midi���̓|�[�g)��ڑ����Ă���
�K�v������܂�:
	% aconnect 64:0 128:0


�Đ��̃��Z�b�g
==============

timidity��SIGHUP�V�O�i���𑗂邱�Ƃɂ��C�Đ����ɂ��ׂẲ����~�߂邱
�Ƃ��ł��܂��B�ڑ��̓��Z�b�g����ێ�����܂����C�C�x���g�͂��͂⏈����
��܂���B�����ĂїL���ɂ���ɂ́C�|�[�g���Đڑ����Ȃ���΂Ȃ�܂���B


���o��
======

������҂�t�@���V�[�Ńr�W���A���ȏo�͂����D�݂ł���΁C�ٍ�̏����ȃv
���O�����Caseqview�����������������B
	% aseqview -p2 &
�����āC(aseqview�ɂ����129:0��129:1���쐬���ꂽ�Ɖ��肵��)timidity�|
�[�g��2�̃|�[�g��ڑ����Ă�������:
	% aconnect 129:0 128:0
	% aconnect 129:1 128:1
�o�͂́C128:0,1�̑����129:0,1�ɓ]������邱�ƂɂȂ�܂��B
	% pmidi -p129:0,129:1 foo.mid


OSS�Ƃ̌݊���
=============

ALSA�V�[�P���T���OSS MIDI�G�~�����[�V�����o�R��timidity�ɃA�N�Z�X����
���Ƃ��ł��܂��B�A�N�Z�X�����f�o�C�X�ԍ��̃`�F�b�N���邽�߁C
/proc/asound/seq/oss�����Ă��������B
       ---------------------------------------
	% cat /proc/asound/seq/oss
	OSS sequencer emulation version 0.1.8
	ALSA client number 63
	ALSA receiver port 0
	...
	midi 1: [TiMidity port 0] ALSA port 128:0
	  capability write / opened none
	
	midi 2: [TiMidity port 1] ALSA port 128:1
	  capability write / opened none
       ---------------------------------------
��L�̏ꍇ���ƁCMIDI�f�o�C�X1��2��timidity�Ɋ��蓖�Ă��Ă��܂��B�����C
playmidi�ŉ��t���Ă݂܂��傤:
	% playmidi -e -D1 foo.mid


�o�O
====

����C����C����ɈႢ�Ȃ��B�B


���\�[�X
========

- ALSA�z�[���y�[�W
	http://www.alsa-project.org
- ����ALSA�n�b�N�y�[�W(aseqview���܂�)
	http://members.tripod.de/iwai/alsa.html
