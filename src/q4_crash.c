#define _GNU_SOURCE
#include "q4.h"

#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

/* Crash capture for the intermittent amdhip64/libstdc++ vector abort.
 *
 * The assert fires inside libamdhip64, the process dies with SIGABRT, and
 * a bare core dump is useless on this box: q4's core would include the
 * ~45 GiB resident expert image, so systemd-coredump drops it ("missing"
 * in coredumpctl).  Two layers here:
 *
 *   1. This handler prints the faulting backtrace straight to stderr
 *      (the q4safe/serve .log captures it) BEFORE the core write — so the
 *      stack survives even when the core does not.
 *   2. The huge weight arenas are marked MADV_DONTDUMP (q4_expert.c) so a
 *      real core stays small enough to store; `coredumpctl debug q4`
 *      then opens it in gdb.
 *
 * backtrace_symbols_fd writes unsymbolized frames but needs no allocation
 * in the fast path; backtrace() itself may resolve through libgcc on first
 * use, so install() pre-warms it. */

static volatile sig_atomic_t q4_in_crash;

static void q4_crash_bt(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    if (q4_in_crash) _exit(128 + sig); /* nested fault: die quietly */
    q4_in_crash = 1;
    char hdr[192];
    int n = snprintf(hdr, sizeof hdr,
                     "\n=== q4 fatal signal %d si_addr=%p pid=%ld ===\n",
                     sig, si ? si->si_addr : NULL, (long)getpid());
    if (n > 0) (void)!write(STDERR_FILENO, hdr, (size_t)n);
    void *fr[64];
    int nf = backtrace(fr, 64);
    backtrace_symbols_fd(fr, nf, STDERR_FILENO);
    /* Re-raise under the default disposition: the kernel then produces a
     * real core through systemd-coredump (the arenas are DONTDUMP).  The
     * signal is blocked inside this handler, so raise() alone would only
     * pend it — unblock first or _exit would win the race and dump
     * nothing. */
    signal(sig, SIG_DFL);
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, sig);
    sigprocmask(SIG_UNBLOCK, &s, NULL);
    raise(sig);
    _exit(128 + sig);
}

void q4_install_crashdump(void) {
    /* Best effort: make sure a core can be written even if the caller's
     * ulimit was 0 (systemd-run inherits the caller's rlimits, but probes
     * launched by hand often run under `ulimit -c 0`). */
    struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_CORE, &rl);
    /* Prewarm the unwinder so the handler never takes its first malloc
     * inside a signal frame. */
    void *fr[8];
    int nf = backtrace(fr, 8);
    char **syms = backtrace_symbols(fr, nf);
    free(syms);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = q4_crash_bt;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    static const int sigs[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE};
    for (size_t i = 0; i < sizeof sigs / sizeof sigs[0]; i++)
        sigaction(sigs[i], &sa, NULL);
}

/* ---- OOM self-containment ----------------------------------------------
 * 2026-10-02 lesson: a bare `./q4` run bypassed serve.sh/q4safe.sh, so the
 * ~45 GiB resident fill put user-1000.slice at ~95% sustained pressure and
 * systemd-oomd killed the Hyprland session unit (and the Devin CLI inside
 * it) — q4 had no leaf cgroup of its own for oomd/the kernel OOM to blame.
 * Every q4 entry point therefore re-execs itself under a dedicated
 * systemd --user scope when no memory cap is already in force — the
 * same scope wrapper serve.sh uses, moved into the binary so no launch path (direct
 * exec, reviewers, ad-hoc scripts) can skip containment.
 *
 * Layers: (a) cgroup MemoryMax + MemorySwapMax=0 makes the kernel kill
 * only this scope on cap overrun; (b) MemoryHigh keeps the scope
 * pressured so systemd-oomd picks our leaf instead of the session slice;
 * (c) oom_score_adj=1000 makes q4 the first victim even when the whole
 * wrap machinery is unavailable. */

static uint64_t q4_scope_cap_bytes(void) {
    const char *cap = getenv("Q4_MEM_MAX");
    uint64_t b;
    if (!cap || !*cap) return 52ull * Q4_GIB;
    /* q4_parse_gib wants a "GB" suffix; accept bare "G" like serve.sh. */
    char nb[40];
    size_t n = strlen(cap);
    snprintf(nb, sizeof nb, "%s%s", cap,
             n && (cap[n - 1] == 'g' || cap[n - 1] == 'G') ? "B" : "");
    if (q4_parse_gib(nb, &b)) return b;
    return 52ull * Q4_GIB;
}

void q4_self_scope(int argc, char **argv) {
    /* Raise our own OOM score first — unprivileged, and covers the case
     * where every other layer below is unavailable. */
    {
        int fd = open("/proc/self/oom_score_adj", O_WRONLY);
        if (fd >= 0) { (void)!write(fd, "1000", 4); close(fd); }
    }
    if (getenv("Q4_CGROUP") || getenv("Q4_NOSCOPE")) return;
    /* Already capped?  A finite memory.max on our leaf cgroup means a
     * wrapper (q4safe/serve/container) already contained us — nesting a
     * second scope adds nothing. */
    {
        char cg[512] = {0};
        FILE *f = fopen("/proc/self/cgroup", "r");
        if (f) {
            char line[512];
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, "0::", 3) == 0) {
                    size_t n = strlen(line + 3);
                    while (n && (line[2 + n] == '\n' || line[2 + n] == '\r'))
                        line[2 + n--] = 0;
                    snprintf(cg, sizeof cg, "%s", line + 3);
                    break;
                }
            }
            fclose(f);
        }
        if (cg[0]) {
            char path[640], val[64] = {0};
            snprintf(path, sizeof path, "/sys/fs/cgroup%s/memory.max", cg);
            f = fopen(path, "r");
            if (f) {
                if (!fgets(val, sizeof val, f)) val[0] = 0;
                fclose(f);
                if (strncmp(val, "max", 3) != 0) return; /* capped leaf */
            }
        }
    }
    /* serve.sh semantics: Q4_MEM_MAX=0 opts out of the cap entirely. */
    {
        const char *cap = getenv("Q4_MEM_MAX");
        if (cap && !strcmp(cap, "0")) return;
    }
    /* systemd --user needs the manager bus.  Bare SSH/agent sessions often
     * lack XDG_RUNTIME_DIR — infer /run/user/<uid> when it exists. */
    {
        const char *xdg = getenv("XDG_RUNTIME_DIR");
        if (!xdg || !*xdg) {
            char rd[64];
            snprintf(rd, sizeof rd, "/run/user/%ld", (long)getuid());
            if (access(rd, F_OK) == 0) setenv("XDG_RUNTIME_DIR", rd, 0);
        }
        const char *bus = getenv("DBUS_SESSION_BUS_ADDRESS");
        if (!bus || !*bus) {
            xdg = getenv("XDG_RUNTIME_DIR");
            char bp[192];
            if (xdg) snprintf(bp, sizeof bp, "%s/bus", xdg);
            if (!xdg || access(bp, F_OK) != 0) {
                fprintf(stderr, "q4: no systemd user bus — running WITHOUT "
                                "a memory cap (oom_score_adj=1000 only)\n");
                return;
            }
        }
    }
    {
        uint64_t maxb = q4_scope_cap_bytes();
        /* High sits just under Max: enough headroom that the ~46 GiB
         * resident image never trips it, pressured enough that oomd sees
         * THIS leaf (not the session slice) as the offender. */
        uint64_t highb = maxb > 2ull * Q4_GIB ? maxb - 2ull * Q4_GIB : maxb;
        char unit[64], vmax[40], vhigh[40];
        snprintf(unit, sizeof unit, "--unit=q4-%ld", (long)getpid());
        snprintf(vmax, sizeof vmax, "MemoryMax=%lluG",
                 (unsigned long long)(maxb / Q4_GIB));
        snprintf(vhigh, sizeof vhigh, "MemoryHigh=%lluG",
                 (unsigned long long)(highb / Q4_GIB));
        const char *nav[24 + argc];
        int i = 0;
        nav[i++] = "systemd-run";
        nav[i++] = "--user";
        nav[i++] = "--scope";
        nav[i++] = "--quiet";
        nav[i++] = "--same-dir";
        nav[i++] = "--collect";
        nav[i++] = unit;
        nav[i++] = "-p"; nav[i++] = vmax;
        nav[i++] = "-p"; nav[i++] = vhigh;
        nav[i++] = "-p"; nav[i++] = "MemorySwapMax=0";
        /* The scope child inherits our full environment — set the marker
         * here instead of interposing `env` (whose exec'ing the literal
         * /proc/self/exe would re-exec env, not us). */
        setenv("Q4_CGROUP", "1", 1);
        static char self[512];
        ssize_t r = readlink("/proc/self/exe", self, sizeof self - 1);
        if (r > 0) self[r] = 0;
        else snprintf(self, sizeof self, "%s", argv[0]);
        nav[i++] = self;
        for (int a = 1; a < argc; a++) nav[i++] = argv[a];
        nav[i] = NULL;
        fprintf(stderr, "q4: no cgroup cap detected — re-exec under "
                        "systemd scope (%s, %s, MemorySwapMax=0)\n",
                vmax, vhigh);
        execvp("systemd-run", (char *const *)nav);
        fprintf(stderr, "q4: systemd-run failed (%s) — running WITHOUT a "
                        "memory cap (oom_score_adj=1000 only)\n",
                strerror(errno));
    }
}
