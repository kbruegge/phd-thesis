\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mimosis}[2017/08/01 Minimal modern thesis class]

\LoadClass[paper=a4,
           twoside,
           pagesize,
           11pt,
           DIV=10,    % TODO: Make configurable
           BCOR=10mm,  % TODO: Make configurable
           cleardoublepage=empty,
           numbers=noenddot,
           titlepage=firstiscover,
           toc=bibliography,
           toc=index,
           captions=tableheading,]{scrbook}

\RequirePackage{ifpdf}
\RequirePackage{ifxetex}
\RequirePackage{ifluatex}

\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\ifxetexorluatex
  \RequirePackage{fontspec}
\else
  \RequirePackage[utf8]{inputenc}
\fi

% Makes it possible to switch between different languages in the text
% while keeping hyphenation rules correct. Should you add another one
% in the list, please ensure that `english` is the last one. The last
% language is used to control standard hyphenation.
\usepackage[ngerman,french,english]{babel}
% /\RequirePackage{polyglossia}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Language
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




\RequirePackage{csquotes} % Context-sensitive quotation marks
\RequirePackage{makeidx}  % For creating indices
\RequirePackage{xspace}   % For automatically "eating" spaces
\RequirePackage[section]{placeins} % keep floats in their section
\RequirePackage{enumitem} % more controll over lists and stuff
\RequirePackage{tabulary}
\RequirePackage{tabularx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-line comments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\comment}[1]{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonts & colours
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[table, usenames,dvipsnames]{xcolor}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\graphicspath{%
  {Figures/}
  {./}
}

% Suppress warnings about page groups in PDFs. This is not justified
% in most of the cases. I am pretty sure I am including my images in
% the right manner.
\begingroup\expandafter\expandafter\expandafter\endgroup
\expandafter\ifx\csname pdfsuppresswarningpagegroup\endcsname\relax
\else
  \pdfsuppresswarningpagegroup=1\relax
\fi

\RequirePackage{subcaption}

% Make sub-references using \subref being typeset with parentheses.
% Otherwise, only the counter will be printed.
\captionsetup{subrefformat=parens}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Glossaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[%
  acronym,
%   automake=immediate,
  automake=true,
  nogroupskip,
  nopostdot,
  nonumberlist,
%   toc,
  ]{glossaries}

% New style that prevents line-breaks between the full description and
% the acronym. Furthermore, it ensures that the acronym is always
% printed in an upright font.
\newacronymstyle{long-short-mimosis}
{%
  \GlsUseAcrEntryDispStyle{long-short}%
}%
{%
  \GlsUseAcrStyleDefs{long-short}%
  \renewcommand*{\genacrfullformat}[2]{%
    \glsentrylong{##1}##2~\textup{(\firstacronymfont{\glsentryshort{##1}})}%
  }%
  \renewcommand*{\Genacrfullformat}[2]{%
    \Glsentrylong{##1}##2~\textup{(\firstacronymfont{\glsentryshort{##1}})}%
  }%
  \renewcommand*{\genplacrfullformat}[2]{%
    \glsentrylongpl{##1}##2~\textup{(\firstacronymfont{\glsentryshortpl{##1}})}%
  }%
  \renewcommand*{\Genplacrfullformat}[2]{%
    \Glsentrylongpl{##1}##2~\textup{(\firstacronymfont{\Glsentryshortpl{##1}})}%
  }%
}

% A new glossary style that based on the long style of the glossaries
% package. It ensures that descriptions and entries are aligned.
\newglossarystyle{long-mimosis}{%
  \setglossarystyle{long}

  \renewcommand{\glossentry}[2]{%
    \glsentryitem{##1}\glstarget{##1}{\glossentryname{##1}} &
    \glossentrydesc{##1}\glspostdescription\space ##2\tabularnewline
  }%
}

% Constrain the description width to enforce breaks.
\setlength{\glsdescwidth}{10cm}

\setacronymstyle{long-short-mimosis}
\setglossarystyle{long-mimosis}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paragraph lists & compact enumerations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[%
    olditem,  % Do not modify itemize environments by default
    oldenum   % Do not modify enumerate environments by default
  ]{paralist}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\KOMAoptions{parskip=half}
\RequirePackage{setspace}
\onehalfspacing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{multirow}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proper typesetting of units
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[binary-units=true]{siunitx}

\sisetup{%
  detect-all           = true,
  detect-family        = true,
  detect-mode          = true,
  detect-shape         = true,
  detect-weight        = true,
  detect-inline-weight = math,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyperlinks & bookmarks & references
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \RequirePackage[%
% colorlinks = true,
% citecolor  = RoyalBlue,
% linkcolor  = RoyalBlue,
% urlcolor   = RoyalBlue,
% ]{hyperref}
\definecolor{mycolor}{HTML}{1f5891}
\definecolor{blueish}{HTML}{545cb7}

\RequirePackage[%
colorlinks = true,
citecolor  = mycolor,
linkcolor  = mycolor,
urlcolor   = mycolor,
]{hyperref}
\urlstyle{same}

\RequirePackage[noabbrev]{cleveref}
\RequirePackage{bookmark}

\crefformat{app}{appendix #2#1#3}
\crefrangeformat{app}{appendices #3#1#4 to #5#2#6}
\crefmultiformat{app}{appendices #2#1#3}{ and #2#1#3}{, #2#1#3}{ and #2#1#3}
\crefrangemultiformat{app}{appendices #3#1#4 to #5#2#6}{ and #3#1#4 to #5#2#6}{, #3#1#4 to #5#2#6}{ and #3#1#4 to #5#2#6}

\Crefformat{app}{Appendix #2#1#3}
\Crefrangeformat{app}{Appendices #3#1#4 to #5#2#6}
\Crefmultiformat{app}{Appendices #2#1#3}{ and #2#1#3}{, #2#1#3}{ and #2#1#3}
\Crefrangemultiformat{app}{Appendices #3#1#4 to #5#2#6}{ and #3#1#4 to #5#2#6}{, #3#1#4 to #5#2#6}{ and #3#1#4 to #5#2#6}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mathematics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{dsfont}
\RequirePackage{mathtools}
\RequirePackage{amssymb}
\RequirePackage{nicefrac}

% Fix the spacing of \left and \right. Use these with the proper bracket
% in order to ensure that they scale automatically.
\let\originalleft\left
\let\originalright\right
\renewcommand{\left}{\mathopen{}\mathclose\bgroup\originalleft}
\renewcommand{\right}{\aftergroup\egroup\originalright}

\DeclareMathOperator*{\argmin}          {arg\,min}
\DeclareMathOperator {\dist}            {dist}
\DeclareMathOperator {\im}              {im}

\newcommand{\domain}{\ensuremath{\mathds{D}}}
\newcommand{\real}  {\ensuremath{\mathds{R}}}

% Proper differential operators
% \newcommand{\diff}[1]{\ensuremath{\operatorname{d}\!{#1}}}
\newcommand{\diff}[1]{\ensuremath{\operatorname{\mathup{d}}\!{#1}}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ordinals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifxetexorluatex
  \newcommand  {\st}{{\addfontfeatures{VerticalPosition=Ordinal}\textup{st}}\xspace}
  \newcommand  {\rd}{{\addfontfeatures{VerticalPosition=Ordinal}\textup{rd}}\xspace}
  \newcommand  {\nd}{{\addfontfeatures{VerticalPosition=Ordinal}\textup{nd}}\xspace}
  \renewcommand{\th}{{\addfontfeatures{VerticalPosition=Ordinal}\textup{th}}\xspace}
  \newcommand{\thraised}{{\addfontfeatures{VerticalPosition=Ordinal}\textup{thraised}}\xspace}
\else
  \newcommand  {\st}{\textsuperscript{\textup{st}}\xspace}
  \newcommand  {\rd}{\textsuperscript{\textup{rd}}\xspace}
  \newcommand  {\nd}{\textsuperscript{\textup{nd}}\xspace}
  \renewcommand{\th}{\textsuperscript{\textup{th}}\xspace}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Penalties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\clubpenalty         = 10000
\widowpenalty        = 10000
\displaywidowpenalty = 10000

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{scrlayer-scrpage}
\pagestyle{scrheadings}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Typefaces for parts, chapters, and sections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\partformat}{\huge\partname~\thepart\autodot}
\renewcommand{\raggedpart}{\flushleft}
\renewcommand{\chapterheadstartvskip}{\vspace{1.5cm}}

% \setkomafont{part}{\normalfont}
\setkomafont{disposition}{\normalfont}
\setkomafont{chapter}{\normalfont\Huge}
\addtokomafont{chapterentry}{\bfseries}
\addtokomafont{partentry}{\bfseries}
% \setkomafont{descriptionlabel}{\normalfont}

\setkomafont{caption}{\small}
\setkomafont{captionlabel}{\usekomafont{caption}}

% Large number for chapter
\renewcommand*{\chapterformat}{%
% \vspace{-10cm}%
%   \begin{minipage}[b]{0.15\textwidth}%
%     % {\color{mycolor}\rule[-5pt]{2pt}{7cm}}%
%     % \hspace{5mm}%
%         {\color{mycolor}\fontsize{50}{55}\selectfont\thechapter\autodot\enskip}%
%   \end{minipage}%
% % 
  {\color{mycolor}\fontsize{50}{55}\selectfont\thechapter\autodot\enskip}
}
