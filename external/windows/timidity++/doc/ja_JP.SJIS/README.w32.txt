TiMidity++ ��Windows�p�ɃR���p�C��������@

�P�DMsys������ Mingw �ŃR���p�C��
�Q�DMsys������ Turbo C++ �ŃR���p�C��
�R�DMsys������ OpenWatcom �ŃR���p�C��
�S�DMsys������ Visual C++ �ŃR���p�C��
�T�DMsys������ Digital Mars �ŃR���p�C��(timiditydrv.dll���R���p�C���ł��Ȃ��j
�U�DMsys������ Pelles C �ŃR���p�C��

�P�DMsys������Mingw�ŃR���p�C��
�i�P�jMingw��MSYS�̃Z�b�g�A�b�v
	�P�jMingw��MSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
	�Q�|�O�jdll�t�@�C������C���|�[�g���C�u�����������@
		�ipexports��mingw-utils-0.3.tar.gz�Ɋ܂܂�Ă���j
    		pexports xxxx.dll >xxxx.def
    		dlltool --dllname xxxx.dll --input-def xxxx.def --output-lib libxxxx.a
	�Q�|�P�jpcurses
		"pdcurses-2.6.0-2003.07.21-1.exe"��Mingw�̃T�C�g����Ƃ��Ă��Đݒ肷��B
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"http://www.vorbis.com/files/1.0.1/windows/OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
		dll�t�@�C������C���|�[�g���C�u����������
    	include\ogg\os_type.h�̂Q�X�s�ڂ�����������
			(os_types.h)
			29 #  if !defined(__GNUC__) || defined(__MINGW32__)
		�N���p�o�b�`�t�@�C���ɃG���g����������
			REM OggVorbis
			set PATH=\usr\local\oggvorbis-win32sdk-1.0.1\bin;\usr\local\oggvorbis-win32sdk-1.0.1\lib;%PATH%
			set C_INCLUDE_PATH=/usr/local/oggvorbis-win32sdk-1.0.1/include:%C_INCLUDE_PATH
			set LD_LIBRARY_PATH=/usr/local/oggvorbis-win32sdk-1.0.1/lib:%LD_LIBRARY_PATH%
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    	dll�t�@�C������C���|�[�g���C�u���������B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
    	�N���p�o�b�`�t�@�C���ɃG���g����������
			REM GOGO
			set PATH=\usr\local\gogo\bin;\usr\local\gogo\lib;%PATH%
			set C_INCLUDE_PATH=/usr/local/gogo/include:%C_INCLUDE_PATH%
			set LD_LIBRARY_PATH=/usr/local/gogo/lib:%LD_LIBRARY_PATH%
	�Q�|4�jflac(http://flac.sourceforge.net/)
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		Change include\*\export.h��58�s�ڂ����ׂĈȉ��̂悤�ɕύX
			(export.h)
			58 #if defined(FLAC__NO_DLL) || !defined(_MSC_VER) \
				|| !defined(__BORLANDC__) || !defined(__CYGWIN32__) || !defined(__MINGW32__)
		�N���p�o�b�`�t�@�C���ɃG���g����������
			REM FLAC
			set PATH=\usr\local\flac-1.2.1-devel-win\lib;;%PATH%
			set C_INCLUDE_PATH=/usr/local/flac-1.2.1-devel-win/include:%C_INCLUDE_PATH%
			set LD_LIBRARY_PATH=/usr/local/flac-1.2.1-devel-win/bin:%LD_LIBRARY_PATH%	
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		portaudio v1.19����Ƃ���portaudio.h��
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		�C���N���[�h�p�X�� portaudio.h ������
		���C�u�����p�X�� portaudio.dll �� libportaudio.dll �Ɩ��O��ς�����������B
		
�i�R�jTiMIdity++�̃R���p�C��
        3-1)timw32g.exe
            (configure)
             CFLAGS="-O2" ./configure --enable-network --enable-w32gui --enable-spline=gauss \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             strip timidity.exe
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
             CFLAGS="-O2" ./configure --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             strip timidity.exe
             mv timidity.exe twsyng.exe

        3-3)twsynsrv.exe
            (configure)
              CFLAGS="-O2" ./configure --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
             add config.h following line
                #define TWSYNSRV 1
           (make)
             make
             strip timidity.exe
             mv timidity.exe twsynsrv.exe

        3-4)timidity.exe
            (configure)
             CFLAGS="-O2" ./configure --enable-interface=ncurses,vt100,winsyn --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             strip timidity.exe
             
        3-5)timiditydrv.dll
            (configure)
             CFLAGS="-O2" ./configure --enable-winsyn --enable-windrv --enable-spline=linear \
             --enable-audio=w32,portaudio
            (make)
             make
             cd windrv
             strip timiditydrv.dll


�Q�DMsys������Turbo C++�ŃR���p�C��
�i�P�jTurbo C++ ��MSYS�̃Z�b�g�A�b�v
	�P�jTurboC++�̃Z�b�g�A�b�v
		
		http://www.codegear.com/jp/downloads/free/turbo ���� Turbo C++ Explorer ���_�E�����[�h���Ă���B
		bcc32.cfg ��ilink 32.cfg������
		(bcc32.cfg)
			-IC:\Borland\BDS\4.0\include
			-LC:\Borland\BDS\4.0\lib
			-LC:\Borland\BDS\4.0\lib\psdk
			-DWINVER=0x0400
			-D_WIN32_WINNT=0x0400
		(ilink32.cfg )
			-LC:\Borland\BDS\4.0\lib;C:\Borland\BDS\4.0\lib\psdk
			
		�i���Ӂjilink32.cfg �Ŏw�肵���p�X�̒��� "-"������� ilink32 ���듮�삷��̂� "_" ���ɕύX����K�v����B
	�Q�jMSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
		"/etc/fstab"����MINGW�̃p�X�̐ݒ�̍s������
		"msys.bat"������������
		(msys.bat�̐擪�s�j
			set PATH=C:\Borland\BDS\4.0\bin;%PATH%
			

�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
	�Q�|�O�|�P�jdll�t�@�C������C���|�[�g���C�u�����������@
		implib -a -c xxx.lib xxx.dll
	�Q�|�O�|�Q�jVC��LIB�t�@�C������BC��LIB�t�@�C���������@
		coff2omf  xxxx.lib xxx_bcpp.lib

	�Q�|�P�jpcurses
		"pdcurses-2.6.0-src.tar.bz2"��Mingw�̃T�C�g����Ƃ��Ă��ăR���p�C������B
		pccurses.lib��libpdcuses.lib�ɖ��O��ς��Ȃ��Ƃ����Ȃ��B
		bcc32.cfg��ilink32.cfg�ɃG���g������������		
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
		dll�t�@�C������C���|�[�g���C�u����������
		bcc32.cfg��ilink32.cfg�ɃG���g������������		
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    	dll�t�@�C������C���|�[�g���C�u���������B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
		bcc32.cfg��ilink32.cfg�ɃG���g������������		
	�Q�|4�jflac(http://flac.sourceforge.net/)
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		Change include\*\export.h��58�s�ڂ����ׂĈȉ��̂悤�ɕύX
			(export.h)
			58 #if defined(FLAC__NO_DLL) || !defined(_MSC_VER) \
				|| !defined(__BORLANDC__) || !defined(__CYGWIN32__) || !defined(__MINGW32__)
		VC��LIB�t�@�C������BC��LIB�t�@�C�������B
		bcc32.cfg��ilink32.cfg�ɃG���g�����������ށB		
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		portaudio v1.19����Ƃ���portaudio.h��
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		�C���N���[�h�p�X�� portaudio.h �������i�w�b�_�t�@�C���������K�v�j

�i�R�jTiMIdity++�̃R���p�C��
        3-0) perl -pe 's/CC\s-o\s\S\S*\s/CC /g' configure >configure_bc
                (configure���� -o xxxx �͂a�b�b���󂯕t���Ȃ�����)
                
        3-1)timw32g.exe
            (configure)
             CC="bcc32" CPP="cpp32" CFLAGS="" ./configure_bc  --enable-w32gui --enable-spline=gauss \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
            CC="bcc32" CPP="cpp32" CFLAGS="" \
            ./configure_bc --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             mv timidity.exe twsyng.exe
        3-3)twsynsrv.exe
            (configure)
             CC="bcc32" CPP="cpp32" CFLAGS=""\
             ./configure_bc --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
             add config.h following line
                #define TWSYNSRV 1
             (make)
             make
             mv timidity.exe twsynsrv.exe
       3-4)timidity.exe
            (configure)
            CC="bcc32" CPP="cpp32" CFLAGS="" \
             ./configure_bc --enable-interface=vt100,winsyn,ncurses --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
       3-5)timiditydrv.dll
            (configure)
            CC="bcc32" CPP="cpp32" CFLAGS="" \
             ./configure_bc  --enable-winsyn --enable-windrv --enable-spline=linear \
             --enable-audio=w32,portaudio
            (make)
             make


�R�DMsys������OpenWatcom�ŃR���p�C��
�i�P�jOpenWatcom��MSYS�̃Z�b�g�A�b�v
	�P�jOpenWatcom�̃Z�b�g�A�b�v(http://www.openwatcom.org)
		�P�|�P�jOpenWatcom���_�E�����[�h���Ă���B
		�Q�|�P�jMicrosoft Platform SDK����rc.exe������Ă��Ă���������B
			�܂��� wine-prgs-0.9.5.zip(http://www.winehq.com/)����wrc.exe���Ƃ肾��rc.exe�Ƀ��l�[������B
		�iOpenWatom��rc.exe�͎g���Ȃ��j
	�Q�jMSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
		"/etc/fstab"����MINGW�̃p�X�̐ݒ�̍s������
		"msys.bat"����"wcc_env.bat"���ĂԂ悤�ɂ���B
		(msys.bat�̐擪�s�j
			call wcc_env.bat
		(wcc_env.bat�̓��e�j
			@echo off
			set LIB=
			set INCLUDE=
			call i:\watcom\setvars.bat
			
			REM OggVorbis
			set PATH=\usr\local\oggvorbis-win32sdk-1.0.1_wcc\bin;\usr\local\oggvorbis-win32sdk-1.0.1_wcc\lib;%PATH%
			set C_INCLUDE_PATH=/usr/local/oggvorbis-win32sdk-1.0.1_wcc/include:%C_INCLUDE_PATH
			set LIB=\usr\local\oggvorbis-win32sdk-1.0.1_wcc\lib:%LIB%
				�i�����j
				 ----
�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
	�Q�|�O�jdll�t�@�C������C���|�[�g���C�u�����������@
		pexports xxxx.dll >xxxx.def
    	lib -def:xxxx.def -machine:IX86
    	pexports�� mingwutils �Ɋ܂܂�Ă���B
	�Q�|�P�jpcurses
		"pdcurses-2.5.0"��GnuWin32(http://sourceforge.net/projects/gnuwin32/)�̃T�C�g����Ƃ��Ă���
		dll�t�@�C������C���|�[�g���C�u���������B
		pccurses.lib��libpdcuses.lib�ɖ��O��ς��Ȃ��Ƃ����Ȃ��B
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
           include\ogg\os_types.h��ҏW����B
              (os_types.h)
              29 #  if defined(__WATCOMC__)
              30 /* MSVC/Borland */
              31 typedef __int64 ogg_int64_t;
              32 typedef int ogg_int32_t;
              33 typedef unsigned int ogg_uint32_t;
              34 typedef short ogg_int16_t;
              35 typedef unsigned short ogg_uint16_t;
              36 #  else
                   -----
              52 #  endif
		dll�t�@�C������C���|�[�g���C�u����������
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    	dll�t�@�C������C���|�[�g���C�u���������B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
		�N���o�b�`�t�@�C���ɃG���g������������		
	�Q�|4�jflac(http://flac.sourceforge.net/)
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		Change include\*\export.h��58�s�ڂ����ׂĈȉ��̂悤�ɕύX
			(export.h)
			58 #if defined(FLAC__NO_DLL) || !defined(_MSC_VER) \
				|| !defined(__BORLANDC__) || !defined(__CYGWIN32__) || !defined(__MINGW32__) \
				|| !defined(__WATCOMC__) || !defined(__DMC__)

		dll�t�@�C������C���|�[�g���C�u����������
		�N���o�b�`�t�@�C���ɃG���g������������
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		�R���p�C���̎d���͂킩��Ȃ���
		portaudio.h���Ƃ肾���B
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		portaudio.dll ����C���|�[�g���C�u���������B
		�N���o�b�`�t�@�C���ɃG���g������������
		�iportaudio.h���������TiMidity++�̓R���p�C���ł���j
�i�R�jTiMIdity++�̃R���p�C��
        3-0)wcc386_w.sh & wpp386_w.sh( scripts/�f�B���N�g���ɂ��� )
            Wcc386 ��GNU��auto Tool�ƒ��������̂Ń��b�p�[���������B
            �p�X�̒ʂ����f�B���N�g���Ɉړ������Ă����āB
            wcc386.exe�̂�����wcc386_w.sh���Ăяo���Ďg���B
        3-1)timw32g.exe
            (configure)
            CC="wcc386_w.sh"  CPP="wcc386_w.sh -p"  CFLAGS="-d0 -obll+riemcht" \
           ./configure --enable-network --enable-w32gui --enable-spline=gauss \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio \
            --disable-oggtest --disable-vorbistest --disable-libFLACtest
            (make)
             make
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
            CC="wcc386_w.sh"  CPP="wcc386_w.sh -p"  CFLAGS="-d0 -obll+riemcht" \
            ./configure --enable-network --enable-winsyng --enable-spline=linear \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio \
            --disable-oggtest --disable-vorbistest --disable-libFLACtest
             (make)
             make
             mv timidity.exe twsyng.exe
        3-3)twsynsrv.exe
            (configure)
            CC="wcc386_w.sh"  CPP="wcc386_w.sh -p"  CFLAGS="-d0 -obll+riemcht" \
            ./configure --enable-network --enable-winsyng --enable-spline=linear \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio \
            --disable-oggtest --disable-vorbistest --disable-libFLACtest
            add config.h following line
                #define TWSYNSRV 1
             (make)
             make
             mv timidity.exe twsynsrv.exe
        3-4)timidity.exe
            (configure)
            CC="wcc386_w.sh"  CPP="wcc386_w.sh -p"  CFLAGS="-d0 -obll+riemcht" \
            ./configure --enable-interface=ncurses,vt100,winsyn --enable-network --enable-spline=linear \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio \
            --disable-oggtest --disable-vorbistest --disable-libFLACtest
            (make)
            make
       3-5)timiditydrv.dll
            (configure)
            CC="wcc386_w.sh"  CPP="wcc386_w.sh -p"  CFLAGS="-d0 -obll+riemcht" \
             ./configure  --enable-winsyn --enable-windrv --enable-spline=linear \
             --enable-audio=w32,portaudio
            (make)
             make

�S�DMsys������Visual C++�ŃR���p�C��
�i�P�jVisual C++��MSYS�̃Z�b�g�A�b�v
	�P�jVisualC++�̃Z�b�g�A�b�v
        Visual C++ 2008 Express Edition(http://www.microsoft.com/japan/msdn/vstudio/express/default.aspx)
		Microsoft Platform SDK
		���_�E�����[�h���ăC���X�g�[������B
	�Q�jMSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
		"/etc/fstab"����MINGW�̃p�X�̐ݒ�̍s������
		"msys.bat"����"vcc_env.bat"���ĂԂ悤�ɂ���B
		(msys.bat�̐擪�s�j
			call vcc_env.bat
		(vcc_env.bat�̓��e�j
			@echo off
			call c:\"Program Files\Microsoft Platform SDK"\SetEnv.Cmd /2000 /RETAIL
			call c:"\Program Files\Microsoft Visual Studio 9.0"\Common7\Tools\vsvars32.bat
			Set INCLUDE=c:\DXSDK\include;%INCLUDE%
			Set LIB=c:\DXSDK\lib;%LIB%
				�i�����j
				 ----
�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
    �Q�|�O�jDLL����C���|�[�g���C�u�����������@
    		pexports xxxx.dll >xxxx.def
    		lib -def:xxxx.def  -machine:x86
    		pexports �� mungwutils �Ɋ܂܂�Ă���B
	�Q�|�P�jpcurses
		"pdcurses-2.6.0-src.tar.bz2"��Mingw�̃T�C�g����Ƃ��Ă��ăR���p�C������B
		pccurses.lib��libpdcuses.lib�ɖ��O��ς��Ȃ��Ƃ����Ȃ��B
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
		�N���o�b�`�t�@�C���ɃG���g������������		
	�Q�|4�jflac(http://flac.sourceforge.net/)
		flac-1.1.0-win.zip���Ƃ��Ă���B
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		�N���o�b�`�t�@�C���ɃG���g������������
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		portaudio.h���Ƃ肾���B
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		portaudio.dll ����C���|�[�g���C�u���������B
		�N���o�b�`�t�@�C���ɃG���g������������
		�iportaudio.h���������TiMidity++�̓R���p�C���ł���j
�i�R�jTiMIdity++�̃R���p�C��
        3-1)timw32g.exe
            (configure)
             CC="cl" CPP="cl.exe -EP"  CFLAGS="-O2" \
            ./configure --enable-network --enable-w32gui --enable-spline=gauss \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
            make
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
            CC="cl" CPP="cl.exe -EP"  CFLAGS="-O2" \
            ./configure --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             mv timidity.exe twsyng.exe
        3-3)twsynsrv.exe
            (configure)
             CC="cl" CPP="cl.exe -EP"  CFLAGS="-O2"\
             ./configure --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
             add config.h following line
                #define TWSYNSRV 1
             (make)
             make
             mv timidity.exe twsynsrv.exe
        3-4)timidity.exe
            (configure)
            CC="cl" CPP="cl.exe -EP"  CFLAGS="-O2" \
             ./configure --enable-interface=ncurses,vt100,winsyn --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
        3-5)timiditydrv.dll
            (configure)
            CC="cl" CXX="cl" CPP="cl.exe -EP"  CFLAGS="-O2" \
              ./configure --enable-winsyn --enable-windrv --enable-spline=linear \
              --enable-audio=w32,portaudio
            (make)
             make

�T�DMsys������Digital Mars�ŃR���p�C��
�i�P�jDigital Mars��MSYS�̃Z�b�g�A�b�v
	�P�jDigital Mars�̃Z�b�g�A�b�v(http://www.digitalmars.com/)
		�P�|�P�jDigital Mars���_�E�����[�h���Ă���B
		�Q�|�P�jMicrosoft Platform SDK����rc.exe������Ă��Ă���������B
			�܂��� wine-prgs-0.9.5.zip(http://www.winehq.com/)����wrc.exe���Ƃ肾��rc.exe�Ƀ��l�[������B
		�iDigital Mars��rcc.exe�͎g���Ȃ��j
	�Q�jMSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
		"/etc/fstab"����MINGW�̃p�X�̐ݒ�̍s������
		"msys.bat"����" dm_env.bat"���ĂԂ悤�ɂ���B
		(msys.bat�̐擪�s�j
			call dm_env.bat
		(dm_env.bat�̓��e�j
			set LIB=
			set INCLUDE=

			Set PATH=i:\dm\bin;%PATH%
			Set INCLUDE=i:\dm\include;i:\dm\include\win32;%INCLUDE%
			Set LIB=i:\dm\lib;%LIB%
			
			Set PATH=i:\usr\local\gogo\bin;%PATH%
			Set INCLUDE=i:\usr\local\gogo\include;%INCLUDE%
			Set LIB=i:\usr\local\gogo\lib;%LIB%
				�i�����j
				 ----
�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
	�Q�|�O�jdll�t�@�C������C���|�[�g���C�u�����������@
		implib out.lib in.dll
	�Q�|�P�jpcurses
		"pdcurses-2.5.0"��GnuWin32(http://sourceforge.net/projects/gnuwin32/)�̃T�C�g����Ƃ��Ă���
		curses.h�̈ȉ��̍s��ύX����B
		281 #if defined( _MSC_VER )|| defined(__DMC__)       /* defined by compiler */
		977 #if !defined(PDC_STATIC_BUILD) && (defined(_MSC_VER) || defined(__DMC__))&& defined(WIN32) && !defined(CURSES_LIBRARY)
		988 # if !defined(PDC_STATIC_BUILD) && (defined(_MSC_VER) || defined(__DMC__)) && defined(WIN32)
		system �I�v�V�������g���āAdll�t�@�C������C���|�[�g���C�u���������B
		$implib /system libpdcurses.lib pdcurses.dll
		pccurses.lib��libpdcuses.lib�ɖ��O��ς��Ȃ��Ƃ����Ȃ��B
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
           include\ogg\os_types.h��ҏW����B
              (os_types.h)
              36 #  elif defined(__MINGW32__) || defined(__DMC__)
		dll�t�@�C������C���|�[�g���C�u����������
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    	dll�t�@�C������C���|�[�g���C�u���������B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
		�N���o�b�`�t�@�C���ɃG���g������������		
	�Q�|4�jflac(http://flac.sourceforge.net/)
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		Change include\*\export.h��58�s�ڂ����ׂĈȉ��̂悤�ɕύX
			(export.h)
			58 #if defined(FLAC__NO_DLL) || !defined(_MSC_VER) \
				|| !defined(__BORLANDC__) || !defined(__CYGWIN32__) || !defined(__MINGW32__) \
				|| !defined(__WATCOMC__) || !defined(__DMC__)
				
		dll�t�@�C������C���|�[�g���C�u����������
		�N���o�b�`�t�@�C���ɃG���g������������
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		portaudio.h���Ƃ肾���B
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		portaudio.dll ����C���|�[�g���C�u���������B
		�N���o�b�`�t�@�C���ɃG���g������������
		�iportaudio.h���������TiMidity++�̓R���p�C���ł���j
�i�R�jTiMIdity++�̃R���p�C��
        3-0-1) unix->dos�̉��s�R�[�h�ϊ��iLF->CRLF)
           $sh script/unix2dos.sh
        3-0-1) perl -pe 's/CC\s-o\s\S\S*\s/CC /g' configure |perl -pe 's/CXX\s-o\s\S\S*\s/CXX /g' - >configure_dm
                (configure���� -o xxxx ��dmc.exe���󂯕t���Ȃ�����)
        3-0-2) Free Pascal Compiler(http://www.freepascal.org/)����
             cpp.exe�����o��fpcpp.exe�Ɩ��O��ς��ăp�X�̒ʂ����f�B���N�g���ɂ����B
            �idmc.exe��gnu autotools�p�̃v���v���Z�b�T�ɂ͕s�����j

        3-1)timw32g.exe
            (configure)
            CC="dmc -Jm -w -mn -5 -o" CPP="fpcpp -D__NT__ -I/i/dm/include" \
             ./configure_dm --enable-network --enable-w32gui --enable-spline=gauss \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
            CC="dmc -Jm -w -mn -5 -o" CPP="fpcpp -D__NT__ -I/i/dm/include" \
             ./configure_dm --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
             (make)
             make
             mv timidity.exe twsyng.exe
        3-3)twsynsrv.exe
            (configure)
            CC="dmc -Jm -w -mn -5 -o" CPP="fpcpp -D__NT__ -I/i/dm/include" \
             ./configure_dm --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            add config.h �̍Ō�Ɉȉ��̍s��ǉ��B
                #define TWSYNSRV 1
             (make)
             make
             mv timidity.exe twsynsrv.exe
        3-4)timidity.exe
            (configure)
            CC="dmc -Jm -w -mn -5 -o" CPP="fpcpp -D__NT__ -I/i/dm/include" \
             ./configure_dm --enable-interface=ncurses,vt100,winsyn --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
            make
       3-5)timiditydrv.dll
            (configure)
            CC="dmc -Jm -w -mn -5 -o" CPP="fpcpp -D__NT__ -I/i/dm/include" \
             ./configure_dm --enable-interface=windrv,winsyn --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
            make
              commentout timiditydrv.h 23:
              23 //#error this stub requires an updated version of <rpcndr.h>
           	make

�U�DMsys������ Pelles C �ŃR���p�C��
�i�P�jPelles C �� MSYS �̃Z�b�g�A�b�v
	�P�jPelles C �̃Z�b�g�A�b�v
		Pelles C �̃z�[���y�[�W�ihttp://www.smorgasbordet.com/pellesc/�j����_�E�����[�h���ăC���X�g�[������B
	�Q�jMSYS�̃Z�b�g�A�b�v(http://sourceforge.net/projects/mingw/�@http://mingw.sourceforge.net/MinGWiki/))
		"/etc/fstab"����MINGW�̃p�X�̐ݒ�̍s������
		"msys.bat"����"pocc_env.bat"���ĂԂ悤�ɂ���B
		(msys.bat�̐擪�s�j
			call pocc_env.bat
		(vcc_env.bat�̓��e�j
			@echo off
			call call c:\PellesC\bin\povars32.bat

			Set INCLUDE=c:\DXSDK\include;%INCLUDE%
			Set LIB=c:\DXSDK\lib;%LIB%
				�i�����j
				 ----
�i�Q�j�g�p����郉�C�u�����B�̃Z�b�g�A�b�v
	�Q�|�P�jpcurses
		"pdcurses-2.6.0-src.tar.bz2"��Mingw�̃T�C�g����Ƃ��Ă��ăR���p�C������B
		pccurses.lib��libpdcuses.lib�ɖ��O��ς��Ȃ��Ƃ����Ȃ��B
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�Q�joggvorbis(http://www.vorbis.com/)
    	"OggVorbis-win32sdk-1.0.1.zip�h���Ƃ��Ă���
    	(http://www.vorbis.com/files/1.0.1/windows/OggVorbis-win32sdk-1.0.1.zip)
		�N���o�b�`�t�@�C���ɃG���g������������		
    �Q�|�R�j�ߌ�̃R�[�_�[(http://www.marinecat.net/mct_top.htm)
    	Gogo.dll ���ߌ�̃R�[�_�[����gogo.h���\�[�X�t�@�C������Ƃ肾���B
    		move gogo.h gogo\include\gogo
    		move gogo.dll libgogo.a gogo\lib
		�N���o�b�`�t�@�C���ɃG���g������������		
	�Q�|4�jflac(http://flac.sourceforge.net/)
		"http://downloads.sourceforge.net/flac/flac-1.2.1-devel-win.zip" ���Ƃ��Ă���B
		Change include\*\export.h��58�s�ڂ����ׂĈȉ��̂悤�ɕύX
			(export.h)
			58 #if defined(FLAC__NO_DLL) || !defined(_MSC_VER) \
				|| !defined(__BORLANDC__) || !defined(__CYGWIN32__) || !defined(__MINGW32__) \
				|| !defined(__WATCOMC__) || !defined(__DMC__)

		�N���o�b�`�t�@�C���ɃG���g������������
	�Q�|�T�jportaudio(http://www.portaudio.com/)
		�R���p�C���̎d���͂킩��Ȃ���		portaudio.h���Ƃ肾���B
		http://sourceforge.net/project/showfiles.php?group_id=81968 �� Csound5.08.2-gnu-win32-f.exe  ���甲���o���� portaudio.dll ��p�ӂ���B
		�N���o�b�`�t�@�C���ɃG���g������������
		�iportaudio.h���������TiMidity++�̓R���p�C���ł���j
		portaudio.h���������TiMidity++�̓R���p�C���ł���B
�i�R�jTiMIdity++�̃R���p�C��
		3-0-1) perl -pe 's/CC\s-o\s\S\S*\s/CC /g' configure >configure_pocc
                (configure���� -o xxxx �� Pelles C ���󂯕t���Ȃ�����)
        3-0-2) �S�Ẵ\�[�X�̊����R�[�h��SJIS�B���s�R�[�h��CR/LF�ɕϊ����Ă����B
        3-1)timw32g.exe
            (configure)
            CC="cc" CPP="pocc.exe -E"  CFLAGS="-MT" ./configure_pocc  \
            --enable-network --enable-w32gui --enable-spline=gauss \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
            make
             mv timidity.exe timw32g.exe
        3-2)twsyng.exe
            (configure)
            CC="cc" CPP="pocc.exe -E"  CFLAGS="-MT" ./configure_pocc  \
            --enable-network --enable-winsyng --enable-spline=linear \
            --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
             mv timidity.exe twsyng.exe
        3-3)twsynsrv.exe
            (configure)
             CC="cc" CPP="pocc.exe -E"  CFLAGS="-MT" ./configure_pocc  \
             --enable-network --enable-winsyng --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
             add config.h following line
                #define TWSYNSRV 1
             (make)
             make
             mv timidity.exe twsynsrv.exe
        3-4)timidity.exe
            (configure)
            CC="cc" CPP="pocc.exe -E"  CFLAGS="-MT" ./configure_pocc  \
            --enable-interface=ncurses,vt100,winsyn --enable-network --enable-spline=linear \
             --enable-audio=w32,vorbis,gogo,ogg,flac,portaudio
            (make)
             make
        3-5)timiditydrv.dll
            (configure)
            CC="cc" CPP="pocc.exe -E"  CFLAGS="-MT" ./configure_pocc  \
            --enable-winsyn --enable-windrv --enable-spline=linear \
              --enable-audio=w32,portaudio
            (make)
             make


2008.4.10 ���i�@�\�i(skeishi@yahoo.co.jp)
