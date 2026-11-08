#!/usr/bin/perl

use warnings;
use strict;

use feature 'signatures';

# convert various Sudoku formats to pencilmarks formats

# parameters:
# -finputspec:outputspec (example -fS:T)

#input formats:
# S: plain 81 cells Sudoku input
#   Unset cells can be represented by '.' or '0'.
#   Otherwise the cells value is given by its digit.
# T: tdoku pencilmarks input
#   Tdoku pencilmarks are represented by 729 bits plus the newline.
#   Each byte represents by its position and value exactly one bit of the board.
#   If the byte is '.', the  bit is not set and the corresponding candidate is
#   not valid. Each cell rYcX is populated by nine bits starting at Y*81+X*9.
#   The nine candidate bits of values '1'-'9', if set to '1'-'9' are set
#   for the corresponding candidate digit.   
# P: px pencilmarks input
#   Pencilmarks (.px) format is a Schoku fixed size text format.
#   Each board is contained in a single line of 2+81*3 bytes, plus the newline.
#   "PX" (2 bytes) identifies the type.
#   Each cell of 9 bits is represented as a hex value of 3 bytes.
# V: variable pencilmarks inputs separating 81 digit strings composed of digits 1-9
#   Variable pencilmarks input:
#    Each puzzle is represented as 81 groups of digits 1-9, which
#    are separated by an arbitrary length sequence (at least 1) of:
#        space, tab, newline, carriage return, single quote('), double quote("), asterisk(*), number sign(#),
#        comma(,), dot(.), minus(-), semicolon(;), colon(:), plus(+), vertical bar(|), tilde(~).
#        Note that neither the digit '0' nor alphabetic characters are allowed
#        in the digit sequence nor in the 'arbitrary' portion.
#    After the 81st group of digits any text (alphabetic characters including 0) are allowed
#        before the next (termination) newline to allow for annotations.
#    The next (puzzle) is scanned for after that termination newline.
#        If further trailing non-numeric characters are present after at the end of the file
#        this is silently ignored.
# C: This is considered a sub-species of V, yet has its own matcher.
#  Each puzzle is represented as 81 groups of digits 1-9, which are joined by ','. 
# B: binary: 81 unsigned short values (little-endian)
#   Each cell is represented in 2 bytes.
#   position.
#   The lowest nine bits represent the nine possible candidates.
#   This is the Schoku native board representation, including the set cell
#   values (only a single bit is set).

#output formats:
# T: tdoku pencilmarks output
# P: px pencilmarks output
# C: comma separated cell strings composed of digits 1-9
# B: binary: 81 unsigned short values

my $filespecs = shift or die "Usage: $0 -finputspec:outputspec\n";

if ( $filespecs eq '-h' ) {
    print '
    -finputspec:outputspec (example -fS:T)

    available inputspecs are:
    P: px pencilmarks input
    S: plain 81 cells Sudoku input
    T: tdoku pencilmarks input
    V: variable pencilmarks inputs separating 81 digit strings composed of digits 1-9

    available outputspecs are:
    B: binary: 81 unsigned short values
    C: comma separated cell strings composed of digits 1-9
    P: px pencilmarks
    T: tdoku pencilmarks
';

}

my $opt = substr($filespecs, 0, 2);
if ( $opt ne '-f' ) { 
    die "Usage: $0 -finputspec:outputspec\n";
}

$filespecs = substr($filespecs, 2); 
my @specs = split (/:/, $filespecs);
my $inspec = $specs[0];
my $outspec = $specs[1];

my $infileh = \*STDIN;
my $outfileh = \*STDOUT;
if ( $inspec eq 'B' ) {
    binmode($infileh);
}
if ( $outspec eq 'B' ) {
    binmode($outfileh);
}


#input subs
my $get_chunk;
my $split_chunk_and_digest;

#output subs
my $format_cells_and_output;

# assign the relevant sub for the input spec provided
if ( index($inspec,'STPC') ) {
    $get_chunk = sub ( $FILE ) {
        chomp(my $line = <$FILE>);
        return $line;
    };
    if ( $inspec eq 'S' ) {
        $split_chunk_and_digest = sub($line) {
            my @board = split //, $line;
            ($#board+1) == 81 || die "Sudoku input spec (S): found " . ($#board+1) . " cells";
            map {
                if ( $_ eq '0' || $_ eq '.') {
                    $_ = 0x1ff;
                } else {
                    $_ = 1<<($_-1);
                }
            } @board;
            return \@board;
        }
    } elsif ($inspec eq 'T' ) {
        $split_chunk_and_digest = sub($line) {
            my @board = ($line =~ /.{9}/g);
            $#board+1 == 81 || die "Sudoku input spec (T): found " . ($#board+1) . " cells";
            map {
                my @chars = split //, $_;
                my $val = 0;
                map {
                    if ( $chars[$_] ne '.') {
                        $chars[$_] == $_+1 || die "Sudoku input spec (T): found invalid candidate value $chars[$_] at position " . ($_+1);
                        $val |= 1<<$_;
                    }
                } (0..8);
                if ( $val == 0 ) {
                    $val = 0x1ff;
                }
                $_ = $val; # $val now has the bit pattern representing the valid candidates
            } @board;
            return \@board;
        }
    } elsif ( $inspec eq 'P' ) {
        $split_chunk_and_digest = sub($line) {
            substr($line,0,2) eq 'PX' || die "Sudoku input spec (P): missing input spec signature prefix of 'PX'";
            $_ = substr($line, 2, length($line)-2);
            my @board = ($_ =~ /.{3}/g);
            $#board+1 == 81 || die "Sudoku input spec (P): found " . ($#board+1) . " cells";
            @board = map {
                $_ = hex($_);
                $_ <= 0x1ff || die "Sudoku input spec (P): found invalid board cell value of $_";
                if ( $_ == 0 ) {
                    $_ = 0x1ff;
                }
                $_
            } @board;
            return \@board;
        }
    } elsif ($inspec eq 'C' ) {
        $split_chunk_and_digest = sub($line) {
            my @board = split /,/, $line;
            ($#board+1) == 81 || die "Sudoku input spec (C): found " . ($#board+1) . " cells";
            map {
                my $val = 0;
                my @chars = split //, $_;
                map {
                    $val |= 1<<($_-1)
                } @chars;
                if ( $_ == 0 ) {
                    $_ = 0x1ff;
                }

                $_ = $val; # $val now has the bit pattern representing the valid candidates
            } @board;
            return \@board;
        }
    }
}

#OK, some caveats regarding input spec 'V':
# we don't really care for line separators to be preserved while
# gobbling up digit groups until we have enough for a board.
# This relies on the rule that the trailing non-digit group ends with a newline.

if ( $inspec eq 'V' ) {
    $get_chunk = sub($FILE) {
        my $chunkcount = 0;
        my @chunks = ();
        my @board = ();
        do {
            my $line = <$FILE>;
            push (@board, split /[^123456789]+/, $line);
            push (@chunks, split /[123456789]+/, $line);
        } while ($#board+1 < 81 && !eof($FILE));
        # validate, then discard useless @chunks
        # cell count of board:
        @board = grep { length($_) } @board;
        if ( @board == 0 ) {
            return '';
        }
        $#board+1 == 81 || die "Sudoku input spec (V): found " . ($#board+1) . " cells";
        # last chunk is arbitrary:
        pop (@chunks) !~ /0/ || die "Sudoku input spec (V): found character '0' in last chunk of data";
        map {
            $_ =~ /^[^\s\n\r'"*#,.-;:+|~[:alnum:]]*$/ ? $_=() : 1; #pass, delete
            // || die "Sudoku input spec (V): found character $_ in chunk of separating characters";
        } @chunks;
        map {
            my $val = 0;
            my @chars = split //, $_;
            map {
                $val |= 1<<($_-1)
            } @chars;
            $val != 0 || die "Sudoku input spec (C): found invalid cell with 0 value";
            $_ = $val; # $val now has the bit pattern representing the valid candidates
        } @board;
        my $packed = pack ('S<*', @board);
        return $packed; 
    };

    $split_chunk_and_digest = sub($packed) {
        my @board = unpack ('S<*', $packed);
        return \@board;
    };
}

if ( $inspec eq 'B' ) {
    binmode($infileh);
    $get_chunk = sub($FILE) {
        my $chunksize = 162;
        my $inpacked;
        my $count = read($FILE, $inpacked, $chunksize);
        if ( $count == $chunksize ) {
            return $inpacked;
        }
    };
    $split_chunk_and_digest = sub {
        my $binaries = $_[0];
        my @board = unpack ('S<*', $binaries);
        $#board+1 == 81 || die "Sudoku input spec (B): found " . ($#board+1) . " cells";
        @board = map {
            if ( $_ == 0 ) {
                $_ = 0x1f;
            }
        }
        return \@board;
    };
}    

# assign the relevant sub for the output spec provided
if ( $outspec eq 'B' ) {
    binmode($outfileh);
    $format_cells_and_output = sub {
        my $boardref = $_[0];
        my $packed = pack ('S<*', @$boardref);
        return $packed; 
    }
} elsif ( $outspec eq 'C' ) {
    $format_cells_and_output = sub($boardref) {
        my $out = join( ',', map {
           my $cell = $_;
           my $digits = '';
           if ( $cell == 0 ) {
               $_ = '123456789';
           } else {
               map {
                   if ( $cell & (1<<$_) ) {
                       $digits .= ($_+1);
                   }
               } (0..8);
               $_ = $digits;
           }
         } @$boardref) . "\n";
         return $out;
    }
}
elsif ( $outspec eq 'P' ) {
    $format_cells_and_output = sub {
        my $boardref = $_[0];
        my $out = 'PX' .
        join( '', map {
                $_ = sprintf("%03x", $_);
            } @$boardref) . "\n";
        return $out;
    }
}
elsif ( $outspec eq 'S' ) {
    $format_cells_and_output = sub {
        my $boardref = $_[0];
        my @out = map {
            my $lsb = $_ & (-$_);
            my $digit = 0;
            if ( $_ == 0 || $_ == 0x1ff || $lsb != $_ ) {
                $digit = 0;
            } else {
                for ( my $k = 9; $k>0; $k-- ) {
                    if ( $lsb == (1<<($k-1)) ) {
                        $digit = $k;
                        last;
                    }
                }
            }
            $_ = $digit . '';
        } @$boardref;
        return join('', @out) . "\n";
    }
} elsif ( $outspec eq 'T' ) {
    $format_cells_and_output = sub {
        my $boardref = $_[0];
        my $out = '';
        map {
            my $i = 1;
            do {
                if ( $_ & 1 ) {
                    $out .= $i;
                } else {
                    $out .= '.';
                }
                $i++;
                $_ = $_ >> 1;
            } while ( $i <= 9 );
        } @$boardref;
        return $out . "\n";
    }
}

do {
    my $chunk = $get_chunk->($infileh);
    if ( length($chunk) ) {
        my $boardref = $split_chunk_and_digest->($chunk);
        print $outfileh $format_cells_and_output->($boardref);
    }

} until ( eof() );
