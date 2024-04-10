%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WIT-THESIS.CLS
%%
%% Last Update: 2022-09-12 by Micah Schuster
%%
%%
%% 2022-09-12 - Removed Signing lines and date of approval on committee page
%% 2022-02-24 - Initial Release Version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{wit-thesis}
[2017/05/08
WIT-Thesis Class for Applied Computer Science]
\RequirePackage{calc}% part of LaTeX tools bundle   
%
% We have to use the Times font for roman family font rather then Computer
% Modern.  The problem being that Computer Modern does not have bold
% smallcaps which are required.  We also use Helvetica for sans serif.  We
% will however use the Computer Modern font for typewriter family.
%
%\renewcommand{\rmdefault}{ptm}  %(Adobe Times)
%\renewcommand{\sfdefault}{phv}  %(Adobe Helvetica)
%\renewcommand{\ttdefault}{cmtt} $(Computer Modern Typewriter)
%\renewcommand{\familydefault}{\sfdefault}

%
% The 'savepaper' option will give small margins,
% single spaced, smaller font and all that.  Useful for
% early printouts
%
\newif\if@issavepaper
\@issavepaperfalse
\DeclareOption{savepaper}{
  \@issavepapertrue
  % FIXME: should we?
  %\PassOptionsToClass{draft}{report}
}

%
% The 'doublespace'
%
\newif\if@isdoublespace
\@isdoublespacefalse
\DeclareOption{doublespace}{
  \@isdoublespacetrue
}

%
% FIXME: We should also support twoside option for twosided printing
%


\DeclareOption{10pt}{
  \@latex@error{WIT-Thesis class does not handle 10pt fonts, only uses 12pt,
don't use the 10pt option}
}

\DeclareOption{11pt}{
  \@latex@error{WIT-Thesis class does not handle 11pt fonts, only uses 12pt,
don't use the 11pt option}
}

\DeclareOption{twoside}{
  \@latex@error{WIT-Thesis class only handles one sided printing,
don't use the twoside option}
}

\DeclareOption{twocolumn}{
  \@latex@error{WIT-Thesis class only handles one column printing,
don't use the twocolumn option}
}

%
% FIXME: what other options do we not support?
%

%%
%% All other options are passed to the base class directly.
%%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

\ProcessOptions

% Report sets up a bunch of stuff for us
\if@issavepaper
  \LoadClass[10pt,oneside,letterpaper]{report}
\else
  % Note: we really depend on 12pt being here, never ever use anything else
  \LoadClass[12pt,oneside,letterpaper]{report}
\fi

%%
%% What to do at begin document
%%
\AtBeginDocument{%
  % If longtable package is used, redefine the caption stuff a bit
  \ifx\LT@c@ption\undefined
  \else
    \typeout{Note: longtable package detected, redefining the caption printing}
    % Must redefine the caption stuff
    \def\LT@c@ption#1[#2]#3{%
      \LT@makecaption#1\fnum@table{#3}%
      \def\@tempa{#2}%
      \ifx\@tempa\@empty\else
         {\let\\\space
         \addcontentsline{lot}{table}%
           {\numberline{\thetable}\ignorespaces #2}}%
      \fi}
    % Code might likely have to use LTcapwidth to make the caption the right
    % size
    \def\LT@makecaption#1#2#3{%
      \LT@mcol\LT@cols l{%
	{ \hspace{-2.5\arraycolsep} %
	  %FIXME: the above hspace is not likely correct, but it works
          \parbox[t]{\LTcapwidth}%
           {\ssp 
            \raggedright \bfseries #1{#2. }#3 \vspace*{\nspextrabaseline} \par}%
        }
      }}%
  \fi
}


%%
%% Utility to comment out parts
%%
\newcommand{\ignore}[1]{}

\renewcommand{\author}[1]%
  {\gdef\@author{#1}}
\renewcommand{\title}[1]%
  {\gdef\@title{#1}}
\newcounter{titlelines}
\setcounter{titlelines}{1}
\newcommand{\titleheading}[1]%
  {\gdef\@titleheading{#1}}
\newcommand{\degree}[1]%
  {\gdef\@degree{#1}}
\newcommand{\degreein}[1]%
  {\gdef\@degreein{#1}}
\newcommand{\degreeONE}[1]%
  {\gdef\@degreeONE{#1}}
\newcommand{\degreeTWO}[1]%
  {\gdef\@degreeTWO{#1}}
\newcommand{\degreeTHREE}[1]%
  {\gdef\@degreeTHREE{#1}}
\newcommand{\switchofto}[1]%
  {\gdef\@switchofto{#1}}
\newcommand{\gradyear}[1]%
  {\gdef\@gradyear{#1}}
\newcommand{\submitdate}[1]%
  {\gdef\@submitdate{#1}}
\newcommand{\thesisname}[1]%
  {\gdef\@thesisname{#1}}

% Comittee Chair
\newcommand{\committeechair}[1]%
  {\gdef\@committeechair{#1}}
\newcommand{\committeechairdept}[1]%
  {\gdef\@committeechairdept{#1}}
%Comittee Co-Char
\newif\ifdocochair
\docochairfalse
  \newcommand{\committeecochair}[1]%
  {\gdef\@committeecochair{#1}\docochairtrue}
\newcommand{\committeecochairdept}[1]%
  {\gdef\@committeecochairdept{#1}}
% Second committee member
\newcommand{\committeesecond}[1]%
  {\gdef\@committeesecond{#1}}
\newcommand{\committeeseconddept}[1]%
  {\gdef\@committeeseconddept{#1}}
% Third committee member
\newcommand{\committeethird}[1]%
  {\gdef\@committeethird{#1}}
\newcommand{\committeethirddept}[1]%
  {\gdef\@committeethirddept{#1}}
% Fourth committee member
\newif\ifdofourthmember
\dofourthmemberfalse
\newcommand{\committeefourth}[1]%
  {\gdef\@committeefourth{#1}
   \dofourthmembertrue}
\newcommand{\committeefourthdept}[1]%
  {\gdef\@committeefourthdept{#1}}

\newcommand{\chapterheader}[1]%
{%
  \addtocontents{toc}{\protect\pagebreak[3]}%
  \addcontentsline{toc}{chaphd}{#1}{}
  \addtocontents{toc}{\protect\nopagebreak}%
}%

%%
%% Here we do chapter headings
%%
\renewcommand{\chaptername}{CHAPTER}
\renewcommand{\appendixname}{APPENDIX}
\newcommand{\appendicesname}{APPENDICES}
\renewcommand{\bibname}{BIBLIOGRAPHY}
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}
% FIXME: OTHERS?


\newif\if@inappendix
\@inappendixfalse
\newif\if@gotappendices
\@gotappendicestrue

\newif\if@infrontmatter
\@infrontmattertrue

\def\@chapter[#1]#2{%
  \if@infrontmatter
    \chapterheader{\chaptername}%
    \setcounter{chapter}{0}%
    \pagenumbering{arabic}%
    \@infrontmatterfalse%
  \fi
  \ifnum \c@secnumdepth >\m@ne
    \refstepcounter{chapter}%
    \typeout{\@chapapp\space\thechapter.}%
    \if@inappendix
      \if@gotappendices
        % same as normal
        \addcontentsline{toc}{chapter}%
          {\protect\numberline{\thechapter}#1}%
      \else
        % no letter
        \addcontentsline{toc}{chapter}{#1}%
      \fi
    \else
      % normal (non-appendix) chapter
      \addcontentsline{toc}{chapter}%
        {\protect\numberline{\thechapter}#1}%
    \fi
  \else
    \addcontentsline{toc}{chapter}{#1}%
  \fi
  \chaptermark{#1}%
  \if@twocolumn
    \@topnewpage[\@makechapterhead{#2}]%
  \else
    \@makechapterhead{#2}%
    % normally there's \@afterheading here but I don't suppose that's
    % needed and it screws with the addvspace command
  \fi%
}


%% make the chapter heading be according to spec
\def\@makechapterhead#1{%
  \if@inappendix
    \if@issavepaper
      \begingroup
        \centering %
        \large\bfseries \@chapapp %
        \if@gotappendices
          \space \thechapter %
        \fi
        \par\nobreak %
      \endgroup
      \vspace*{12pt} %
      \vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
    \else
      % Appendix is supposed to have it's own cover page
      %\vspace*{3in}  %
      \vspace*{\fill} %
      %\par
      \begingroup
        \centering %
        \large\bfseries \@chapapp %
        \if@gotappendices
          \space \thechapter %
        \fi
        \par\nobreak %
	\vspace*{12pt} %
	\vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
	% The parbox constrains the width of the title to parameters
        \parbox[t]{4.5in}{ \noindent \centering \noindent \large\scshape\bfseries #1 }
        \par
        \vspace*{12pt} %
      \endgroup
      \vspace*{\fill}
      \newpage
    \fi
  \fi
  {
    \if@issavepaper\else
      \vspace*{0.5in} %
      \vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
    \fi
    \par
    \centering %
    \parindent \z@ \large %
    \if@inappendix
      % Only do this in savepaper mode, in final mode
      % we are adding a whole page, see above 
      \if@issavepaper
        { \bfseries \@chapapp %
          \if@gotappendices
            \space \thechapter %
          \fi
        } %
        \par\nobreak %
	\vspace*{12pt} %
	\vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
      \fi
    \else
      % Add CHAPTER # heading
      \ifnum \c@secnumdepth >\m@ne
        { \bfseries \@chapapp\space \thechapter } %
        \par\nobreak %
	\vspace*{12pt}
	\vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
      \fi
    \fi
    \par
    % Add the chapter title
    \interlinepenalty\@M
    % The parbox constrains the width of the title to parameters
    \parbox[t]{4.5in}{ \centering \noindent \large\scshape\bfseries %
                       #1 %
                       \par\vspace*{\nspextrabaseline} \vspace*{\nspextrabaselinelg} }%
    \par
    \vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
    \addvspace{12pt}
  }%
  }	

%% make the chapter heading be according to spec
\def\@makeschapterhead#1{%
  { \centering
    \parindent \z@ \normalfont
    \par\nobreak
    % The parbox constrains the width of the title to parameters
    \parbox[t]{4.5in}{ \ssp \centering \noindent \large\scshape\bfseries %
                       #1 %
                       \par\vspace*{12pt} \vspace*{\nspextrabaselinelg} \par }%
    \par
    \vspace*{\nspextrabaseline}
  }%
  }	

%% Renew the appendix command
\renewcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}%
  \@inappendixtrue%
  \@gotappendicesfalse%
  \chapterheader{\appendixname}%
}

%% New \appendices command for multiple appendices
\newcommand\appendices{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}%
  \@inappendixtrue%
  \@gotappendicestrue%
  \chapterheader{\appendicesname}%
}

\setcounter{secnumdepth}{5}

\newcounter {subsubsubsection}[subsubsection]
\renewcommand\thesubsubsubsection{\thesubsubsection .\@arabic\c@subsubsubsection}
\@newctr{paragraph}[subsubsubsection]
\renewcommand\theparagraph{\thesubsubsubsection.\@arabic\c@paragraph}


%
% We never use subparagraphs
%



%
% Section
%
\renewcommand{\section}{\addvspace{12pt} \pagebreak[3] \secdef\@mysect\@myssect}
\newcommand{\@mysect}[2][?]{%
  \refstepcounter{section}%
  \if@inappendix\else
  \addcontentsline{toc}{section}%
     {\protect\numberline{\thesection}#1}%
  \fi
  \typeout{ Section \thesection.}%
  {\centering  \noindent
    \parbox[t]{4.5in}{ \ssp \centering \noindent \large\scshape\bfseries %
	    	       \thesection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}
\newcommand{\@myssect}[2][?]{%
  {\centering \noindent
    \parbox[t]{4.5in}{ \ssp \centering \noindent \large\scshape\bfseries %
	    	       \thesection \space ~ %
                       #2 % 
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}

%
% Subsection
%
\renewcommand{\subsection}{\addvspace{12pt} \pagebreak[3] \secdef\@mysubsect\@myssubsect}
\newcommand{\@mysubsect}[2][?]{%
  \refstepcounter{subsection}%
  \if@inappendix\else
  \addcontentsline{toc}{subsection}%
    {\protect\numberline{\thesubsection}#1}%
  \fi
  \typeout{  Subsection \thesubsection.}%
  {\centering \noindent
    \parbox[t]{4.5in}{ \ssp \centering \noindent \large\bfseries %
	    	       \thesubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}
\newcommand{\@myssubsect}[2][?]{%
  {\centering \noindent
    \parbox[t]{4.5in}{ \ssp \centering \noindent \large\bfseries %
	    	       \thesubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}

%
% Subsubsection
%
\renewcommand{\subsubsection}{\addvspace{12pt} \pagebreak[3] \secdef\@mysubsubsect\@myssubsubsect}
\newcommand{\@mysubsubsect}[2][?]{%
  \refstepcounter{subsubsection}%
  \if@inappendix\else
  \addcontentsline{toc}{subsubsection}%
    {\protect\numberline{\thesubsubsection}#1}%
  \fi
  \typeout{   Subsubsection \thesubsubsection.}%
  { \noindent
    \parbox[t]{3.5in}{ \ssp \raggedright \noindent \large\scshape\bfseries %
	    	       \thesubsubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}
\newcommand{\@myssubsubsect}[2][?]{%
  { \noindent
    \parbox[t]{3.5in}{ \ssp \raggedright \noindent \large\scshape\bfseries %
	    	       \thesubsubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}

%
% Subsubsubsection
%
\newcommand{\subsubsubsection}{\addvspace{12pt} \pagebreak[3] \secdef\@mysubsubsubsect\@myssubsubsubsect}
\newcommand{\@mysubsubsubsect}[2][?]{%
  \refstepcounter{subsubsubsection}%
  \if@inappendix\else
  \addcontentsline{toc}{subsubsubsection}%
    {\protect\numberline{\thesubsubsubsection}#1}%
  \fi
  \typeout{    Subsubsubsection \thesubsubsubsection.}%
  { \noindent
    \parbox[t]{3.5in}{ \ssp \raggedright \noindent \large\bfseries %
	    	       \thesubsubsubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}
\newcommand{\@myssubsubsubsect}[2][?]{%
  { \noindent
    \parbox[t]{3.5in}{ \ssp \raggedright \noindent \large\bfseries %
	    	       \thesubsubsubsection \space ~ %
                       #2 %
                       \par \nsp \par \vspace*{\nspextrabaselinelg} }%
  \par\nopagebreak}%
}

\renewcommand\paragraph{\@startsection{paragraph}{5}{\z@}%
                                    {12pt}%
                                    {-\fontdimen2\font plus -\fontdimen3\font minus -\fontdimen4\font}%
                                    {\normalfont\normalsize\bfseries}}


%
% Coversheet for an oversized figure
%
\newcommand{\figurecoversheet}[1]{
  \begingroup
    \clearpage
    \vspace*{\fill}
    \par\noindent
    \def\@captype{figure}
    \caption{#1}
    \par\vspace*{\fill}
    \newpage
  \endgroup
}


% table already defined in the report.cls
% we just change a few things, like make things
% single spaced inside etc
\renewenvironment{table}[1][hbt]
                 {\@float{table}[#1]
		  \setlength\abovecaptionskip{0pt}
		  \setlength\belowcaptionskip{\nspextrabaseline}
		  \ssp %float has it's own scope
		 }
                 {\end@float}
\renewenvironment{table*}[1][hbt]
                 {\@dblfloat{table}[#1]
		  \setlength\abovecaptionskip{0pt}
		  \setlength\belowcaptionskip{\nspextrabaseline}
		  \ssp %float has it's own scope
		 }
                 {\end@dblfloat}

\renewenvironment{figure}[1][hbt]
                 {\@float{figure}[#1]
		  \setlength\abovecaptionskip{\nspextrabaseline}
		  \setlength\belowcaptionskip{0pt}
		  \ssp %float has it's own scope
		 }
                 {\end@float}
\renewenvironment{figure*}[1][hbt]
                 {\@dblfloat{figure}[#1]
		  \setlength\abovecaptionskip{\nspextrabaseline}
		  \setlength\belowcaptionskip{0pt}
		  \ssp %float has it's own scope
		 }
                 {\end@dblfloat}

% 12pt + 8pt is about the 1.5 spacing + 12pt
\floatsep 20pt
\dblfloatsep 20pt
\intextsep 20pt
\textfloatsep 20pt
\dbltextfloatsep 20pt

% Table/Figure captions have different formatting
% from the standard report class.  Note that it's far
% from perfect but I don't know how to make it conform
% to guidelines automatically especially for small
% figures/tables
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  {\ssp \raggedright \bfseries #1. #2\par}
  \vskip\belowcaptionskip}

% We must redefine the whole caption thing to make the list of tables/figures
% stuff work correctly
\long\def\@caption#1[#2]#3{%
  \par
  \addcontentsline{\csname ext@#1\endcsname}{#1}%
    {\csname fnum@#1\endcsname.\space\ignorespaces #2}%
  \begingroup
    \@parboxrestore
    \if@minipage
      \@setminipage
    \fi
    \normalsize
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

%%
%% Single and normal (1.5) linespacing
%%
\def\ssp{\def\baselinestretch{1.0}\large\normalsize}
% For 12pt (\normalsize) (baselineskip is 14.5pt)
\def\sspextrabaseline{2.5pt}%
% For 14pt (\large) (baselineskip is 18pt)
\def\sspextrabaselinelg{4pt}%

% normal spacing
\def\nsp{\def\baselinestretch{1.24}\large\normalsize}
% For 12pt (\normalsize) (baselineskip is 14.5pt)
\def\nspextrabaseline{5.98pt}%
% For 14pt (\large) (baselineskip is 18pt)
\def\nspextrabaselinelg{8.32pt}%

% Fully double spaced, used for the titlepage
\def\dsp{\def\baselinestretch{1.66}\large\normalsize}
% For 12pt (\normalsize) (baselineskip is 14.5pt)
\def\dspextrabaseline{12.07pt}%
% Note that for large this is a bit too much, but still ok,
% there's not that much large text around
% For 14pt (\large) (baselineskip is 18pt)
\def\dspextrabaselinelg{15.88pt}%

% This selects normal spacing to be whatever
% we want to (either \ssp or \dsp)
\if@isdoublespace
  \let\nsp\dsp
  \def\nspextrabaseline{\dspextrabaseline}%
  \def\nspextrabaselinelg{\dspextrabaselinelg}%
\fi
\if@issavepaper
  \let\nsp\ssp
  \def\nspextrabaseline{\sspextrabaseline}%
  \def\nspextrabaselinelg{\sspextrabaselinelg}%
\fi


%
% Switch to normal spacing
%
\nsp

\topmargin -0.375in
% if in fact the fontsize is 12pt, which we assume
\headheight 12pt
\headsep 0in
\advance \headsep by -\topmargin
\advance \headsep by -\headheight
\topskip 0pt

\footskip 30pt

\parskip 0pt

\if@issavepaper
  % Whack margins for the savepaper options to full page
  \topmargin -0.5in %0pt
  \advance \topmargin by -\headheight
  \advance \topmargin by -\headsep
     
  \textheight 8.9in
     
  \oddsidemargin 0pt
  \evensidemargin 0pt
     
  \textwidth 6.5in
\else
  % Non-savepaper real margins
  \oddsidemargin 0.25in
  \evensidemargin 0in    %   Note that \oddsidemargin = \evensidemargin

  % Calculate textheight properly
  %\textheight 8.75in
  \textheight 10in % 1in is the top offset
  \advance \textheight -\topmargin
  \advance \textheight -\headheight
  \advance \textheight -\headsep
  \advance \textheight -1in % bottom margin
  \textwidth 6.25truein
\fi

\if@issavepaper
  \def\ps@plain{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage DRAFT \hfil}%
      \def\@oddhead{\hfil DRAFT \thepage}%
      }
\else
  \def\ps@plain{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage \hfil}%
      \def\@oddhead{\hfil \thepage}%
      }
\fi

%%
%% Our environments and frontmatter stuff
%%

\renewenvironment{thebibliography}[1]%
{
  \@frontmattertitle{\bibname}%
  \addcontentsline{toc}{schapter}{\bibname}
  \begingroup
  \ssp
  \list{\@biblabel{\@arabic\c@enumiv}}%
       {\settowidth\labelwidth{\@biblabel{#1}}%
        \leftmargin\labelwidth
	%this makes 1.5 spaced like entries but without 1.5 spacing
	% the entire thing as the manual requires.
        \parsep\z@%
	\itemsep\nspextrabaseline%
        \advance\leftmargin\labelsep
        \@openbib@code
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m%
}{%
  \def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist%
  \endgroup
  \newpage
}

%
% The title page
%
\renewcommand{\maketitle}%
{%
  %default for length 1
  \def\@wittopskip{1.125in}%
  \def\@wittitlelen{0.8125in}%
  \def\@witthesislen{2.375in}%
  \def\@witdegreelen{2.8125in}%
  \def\@witbottomskip{0.375in}%
  \ifnum \c@titlelines = 2%
    \def\@wittopskip{0.75in}%
    \def\@wittitlelen{1.1875in}%
    \def\@witthesislen{2.375in}%
    \def\@witdegreelen{2.8125in}%
    \def\@witbottomskip{0.375in}%
  \fi
  \ifnum \c@titlelines = 3%
    \def\@wittopskip{0.4375in}%
    \def\@wittitlelen{1.5625in}%
    \def\@witthesislen{2.375in}%
    \def\@witdegreelen{2.8125in}%
    \def\@witbottomskip{0.375in}%
  \fi
  \ifnum \c@titlelines = 4%
    \def\@wittopskip{0.125in}%
    \def\@wittitlelen{2in}%
    \def\@witthesislen{2.375in}%
    \def\@witdegreelen{2.8125in}%
    \def\@witbottomskip{0.25in}%
  \fi
  \thispagestyle{empty}%
  \begingroup%
  \raggedbottom%
  \centering%
  \ssp%
  \vspace*{\@wittopskip}\par%
    \noindent%
    \begin{minipage}[t][\@wittitlelen][t]{\textwidth}%
      \dsp \noindent \centering%
        { \bfseries \@title }
      \par
    \end{minipage} \par%
%%%%%%%%%
  \rule{1.25in}{1pt}\vspace*{-\baselineskip}\vspace*{-1pt}\par%
%%%%%%%%%
    \noindent%
    \begin{minipage}[t][\@witthesislen][c]{\textwidth}%
      \dsp \noindent \centering%
\if@issavepaper
	A {\em DRAFT} of a {\@thesisname} \\
\else
        A {\@thesisname} \\
\fi
        Presented to the \\
        Faculty of \\
        Wentworth Institute of Technology
      \par
    \end{minipage} \par%
%%%%%%%%%
  \rule{1.25in}{1pt}\vspace*{-\baselineskip}\vspace*{-1pt}\par%
%%%%%%%%%
    \noindent%
    \begin{minipage}[t][\@witdegreelen][c]{\textwidth}%
      \dsp \noindent \centering%
        In Partial Fulfillment \\
        of the Requirements for the Degree \\
%        {\@degree} \\
%        {\@switchofto} \\
%        {\@degreein}
        {\@degreeONE} \\
        {\@degreeTWO} \\
        {\@degreeTHREE} \\
      \par%
    \end{minipage} \par%
%%%%%%%%%
  \rule{1.25in}{1pt}\par%
  %no baseline skip since next is just a vfill
%%%%%%%%%
  \vspace*{\fill}\par
  \noindent
  { \dsp
        by \\
        {\@author}\\
        {\@submitdate}
      \par
  }
  \vspace*{\@witbottomskip}
  \endgroup
  \newpage
}%

%
% The signature page
%
\newcommand{\makesignature}%
{%
\if@issavepaper\else
  \thispagestyle{empty}
  \begingroup
  \centering%
  \raggedbottom%
  \setlength\fboxsep{0pt}%
  \ssp%
    \noindent%
    %\rule{12pt}{3.875in}%
    \begin{minipage}[t][3.875in][t]{\textwidth}%
      \dsp \noindent \centering%
      { \large \bfseries WENTWORTH INSTITUTE OF TECHNOLOGY }
      \par\vspace*{\baselineskip}
      The Undersigned Committee Approves the \\
      {\@thesisname} of {\@author}:
      \par\vspace*{\baselineskip}
      {\@titleheading}
      \par
    \end{minipage} \par%
%%%%%%%%%
% Signatures
  %\vspace*{-
  %\rule{3.75in}{1pt}\\
  {\@committeechair}, Chair \\
  {\@committeechairdept}
  \par\vspace*{2\baselineskip}
  %
  \ifdocochair
    %\rule{3.75in}{1pt}\\
    {\@committeecochair}  \\
    {\@committeecochairdept}
    \par\vspace*{2\baselineskip}
  \fi
%
  %\rule{3.75in}{1pt}\\
  {\@committeesecond} \\
  {\@committeeseconddept}
  \par\vspace*{2\baselineskip}
%
  %\rule{3.75in}{1pt}\\
  {\@committeethird} \\
  {\@committeethirddept}
  \par\vspace*{2\baselineskip}
%
  \ifdofourthmember
    %\rule{3.75in}{1pt}\\
    {\@committeefourth}  \\
    {\@committeefourthdept}
  \fi
%%%%%%%%%
% Date
  \par\vspace*{\fill}%
  %  \rule{2.5in}{1pt}\\
  %  Approval Date
  %\par\vspace*{0.125in}
  \endgroup
  \newpage
\fi
}%

\newcommand{\@frontmattertitle}[1]{%
  \clearpage
  \if@issavepaper\else
    \vspace*{0.5in}%
    \vspace*{-\nspextrabaseline}% (baselinestretch-1.0 * baselineskip)
  \fi
  \par%
  {%
    \centering \noindent
    {\large \bfseries #1}
    \vspace*{12pt}%
    \vspace*{-\nspextrabaselinelg}% (baselinestretch-1.0 * baselineskip)
    \par
  }%
}

% make the abstract page
\renewenvironment{abstract}%
{%
  \clearpage
  \ssp %
  \addcontentsline{toc}{schapter}{ABSTRACT}
  \vspace*{0.5in}%
  \par
  \begingroup
    \centering
    {\large\bfseries ABSTRACT OF THE THESIS}
    \par \vspace*{12pt}
    {\@titleheading} \\
    by \\
    {\@author} \\
%    {\@degree} {\@switchofto} {\@degreein} \\
    {\@degreeONE} {\@degreeTWO} {\@degreeTHREE} \\
    Wentworth Institute of Technology, {\@gradyear}
    \par
  \endgroup
  \vspace*{12pt}
}{%
  \par
  \nsp
  \newpage
}%

% make the library abstract page
\newenvironment{libraryabstract}%
{%
  \clearpage
  \oddsidemargin 0pt
  \evensidemargin 0pt
  \pagestyle{empty}
  \thispagestyle{empty}
  \ssp %
  \begingroup
    \centering
    ABSTRACT OF THE THESIS
    \par \vspace*{12pt}
    {\@titleheading} \\
    by \\
    {\@author} \\
%    {\@degree} {\@switchofto} {\@degreein} \\
    {\@degreeONE} {\@degreeTWO} {\@degreeTHREE} \\
    San Diego State University, {\@gradyear}
    \par
  \endgroup
  \vspace*{12pt}
}{%
  \par
  \nsp
  \pagestyle{plain}
  \newpage
}%

% make the acknowledgments page
\newenvironment{acknowledgments}%
{%
  \addcontentsline{toc}{schapter}{ACKNOWLEDGMENTS}
  \@frontmattertitle{ACKNOWLEDGMENTS}
  \begingroup
}{%
  \par
  \endgroup
  \newpage
}%

% make the dedication page
\newenvironment{dedication}%
{%
  \@frontmattertitle{DEDICATION}
  \begingroup
}{%
  \par
  \endgroup
  \newpage
}%

%(GLOSSARY)
%(EXPERIMENTAL)
% make the glossary page
\newenvironment{glossarypage}%
{%
  \addcontentsline{toc}{schapter}{GLOSSARY}
  \@frontmattertitle{GLOSSARY}
  \begingroup
}{%
  \par
  \endgroup
  \newpage
}%


% make the epigraph page
% FIXME: We don't yet support epigraph in chapters, only this
% frontmatter epigraph page
\newenvironment{epigraph}%
{%
  \vspace*{\stretch{3}}
  \begingroup
  \ssp
}{%
  \par
  \endgroup
  \vspace*{\stretch{4}}
  \newpage
}%

% make the copyright page
\newenvironment{copyrightpage}%
{%
  \vspace*{\stretch{3}}
  \begingroup
    \centering
}{%
    \par
  \endgroup
  \vspace*{\stretch{4}}
  \newpage
}%

% Table of contents
\renewcommand{\tableofcontents}{%
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \begingroup
    \@frontmattertitle{\contentsname}%
    \noindent \hfill PAGE
    \par
    \ssp % so that single entries are singlespaced
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
  \endgroup
  \newpage
}
% This makes the right margin enough to get the text out of the space of
% the PAGE header as per manual
\renewcommand{\@tocrmarg}{1.0in}%
\renewcommand{\@dotsep}{1}%
%
% The two lengths in the definition of \l@subsection should sum up the
% first one in the definition of \l@subsubsection; the first lenght is
% the indentation of numbers in the TOC; the second length is the
% space reserved for the numbers.
%
% adding \nspextrabaseline will be like making things 1.5 spaced
\newcommand*\l@chaphd[2]{\vspace{\nspextrabaseline}\noindent #1}
\newcommand*\l@schapter{\vspace{\nspextrabaseline} \@dottedtocline{0}{0pt}{0.25in}}
\renewcommand*\l@chapter{\vspace{\nspextrabaseline} \@dottedtocline{0}{0.25in}{0.25in}}
\renewcommand*\l@section{\ifnum 1>\c@tocdepth\else\vspace{\nspextrabaseline}\fi\@dottedtocline{1}{0.5in}{2.5em}}
\renewcommand*\l@subsection{\ifnum 2>\c@tocdepth\else\vspace{\nspextrabaseline}\fi\@dottedtocline{2}{0.75in}{3em}}
\renewcommand*\l@subsubsection{\ifnum 3>\c@tocdepth\else\vspace{\nspextrabaseline}\fi\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@subsubsubsection{\ifnum 4>\c@tocdepth\else\vspace{\nspextrabaseline}\fi\@dottedtocline{4}{7.0em}{4.1em}}
\renewcommand*\l@paragraph{\ifnum 5>\c@tocdepth\else\vspace{\nspextrabaseline}\fi\@dottedtocline{5}{10em}{5em}}

\setcounter{tocdepth}{2}

% List of tables and figures
\renewcommand{\listoffigures}{%
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \begingroup
    \@frontmattertitle{\listfigurename}%
    \addcontentsline{toc}{schapter}{\listfigurename}
    \noindent \hfill PAGE
    \par
    \ssp % so that single entries are singlespaced
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
  \endgroup
  \newpage
}
\renewcommand{\listoftables}{%
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \begingroup
    \@frontmattertitle{\listtablename}%
    \addcontentsline{toc}{schapter}{\listtablename}
    \noindent \hfill PAGE
    \par
    \ssp % so that single entries are singlespaced
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
  \endgroup
  \newpage
}

% There is no number, we print the table/figure inside the
% name as per the manual
\renewcommand*\l@figure{%
 \vspace{\nspextrabaseline}%
 \@dottedtocline{1}{0pt}{0.5in}%
}
\renewcommand*\l@table{%
 \vspace{\nspextrabaseline}%
 \@dottedtocline{1}{0pt}{0.5in}%
}


% Quotes are singlespaced as per manual
\renewenvironment{quote}
                 {\list{}{\rightmargin\z@%
                          \leftmargin 0.75in}%
                  \item\relax \ssp}
                 {\endlist}
\renewenvironment{verse}
                 {\let\\\@centercr
                  \list{}{\itemsep      \z@
                          \itemindent   -1.5em%
                          \listparindent\itemindent
                          \rightmargin  \z@
			  \leftmargin 0.75in}%
                  \item\relax \ssp}
                 {\endlist}


% Lists are singlespaced with entries 1.5 spaced as per manual
% This is stolen from real latex and modified, thus looks more macho
% then it really is
\def\enumerate{%
  \ifnum \@enumdepth >\thr@@\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \expandafter
      \list
        \csname label\@enumctr\endcsname
        {\itemsep\nspextrabaseline%
         \parsep\z@%
         \partopsep 12pt%
         \advance \partopsep -\nspextrabaseline%
         \topsep \nspextrabaseline%
         % we can't just \ssp here since that might \ssp the
         % preceding paragraph, we must \ssp inside the item
         \def\item{%
           \@inmatherr\item
           \ssp%
           \@ifnextchar [\@item{\@noitemargtrue \@item[\@itemlabel]}}
         \usecounter\@enumctr\def\makelabel##1{\hss\llap{##1}}}%
  \fi}
\let\endenumerate =\endlist
\def\itemize{%
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {\itemsep\nspextrabaseline%
       \parsep\z@%
       \partopsep 12pt%
       \advance \partopsep -\nspextrabaseline%
       \topsep \nspextrabaseline%
       % we can't just \ssp here since that might \ssp the
       % preceding paragraph, we must \ssp inside the item
       \def\item{%
         \@inmatherr\item
         \ssp%
         \@ifnextchar [\@item{\@noitemargtrue \@item[\@itemlabel]}}
       \def\makelabel##1{\hss\llap{##1}}}%
  \fi}
\let\enditemize =\endlist


\renewcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}%
    % single space things, but don't use \ssp, that makes
    % font size normal!
    \def\baselinestretch{1.0}\normalsize\footnotesize%
    #1}

% indent even the first paragraph, we just make sure that @afterindent is
% always true
\let\@afterindentfalse\@afterindenttrue
\@afterindenttrue

\pagenumbering{roman}%
\onecolumn%
\raggedbottom%
% Make text ragged right, no justification as per manual.
% Ugly, harder to read, but that's what's required.
\setlength{\rightskip}{0pt plus 1fil}%
% Make lists ragged right, no justification as per manual
\setlength{\@rightskip}{0pt plus 1fil}%
% half inch indentation as per thesis manual
\parindent 0.5in
\pagestyle{plain}%
% TeX justification 
\sloppy

% Make club/widow lines not happen as per style guide
\clubpenalty \@M    % 'Club line'  at bottom of page.
\widowpenalty \@M   % 'Widow line' at top of page.


% Defaults
\title{PLEASE USE $\backslash$title AND SET titlelines COUNTER}
\titleheading{Please Use $\backslash$titleheading to Set Title}
\author{Please Use $\backslash$author}
\thesisname{Thesis}
\degree{Please Use $\backslash$degree}
\switchofto{in}
\degreein{Please Use $\backslash$degreein}
\gradyear{Please Use $\backslash$gradyear}
\submitdate{Please Use $\backslash$submitdate}
\committeechair{Please Use $\backslash$committeechair}
\committeechairdept{Please Use $\backslash$committeechairdept}
\committeechair{Please Use $\backslash$committeecochair}
\committeechairdept{Please Use $\backslash$committeecochairdept}
\committeesecond{Please Use $\backslash$committeesecond}
\committeeseconddept{Please Use $\backslash$committeeseconddept}
\committeethird{Please Use $\backslash$committeethird}
\committeethirddept{Please Use $\backslash$committeethirddept}
\gdef\@committeefourth{}
\gdef\@committeefourthdept{}

