use Cwd qw(abs_path);
use File::Path qw(make_path);

my $build_root = abs_path('.');
make_path("$build_root/.texmf-var", "$build_root/.mpl-cache");
$ENV{'TEXMFVAR'} = "$build_root/.texmf-var";
$ENV{'TEXMFCACHE'} = "$build_root/.texmf-var";
$ENV{'MPLCONFIGDIR'} = "$build_root/.mpl-cache";

my $python = -x '../../.venv/bin/python' ? '../../.venv/bin/python' : 'python3';
system("$python figures/generate_figures.py") == 0
  or die "Figure generation failed\n";

$lualatex = 'lualatex -interaction=nonstopmode -halt-on-error %O %S';
$pdf_mode = 4;
