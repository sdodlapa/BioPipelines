#!/usr/bin/env python3
"""
Analyze package usage across all container definitions to identify
common packages that should be moved to the base container.
"""
import re
from collections import Counter
from pathlib import Path

def parse_container_packages():
    """Extract packages from all container definitions."""
    containers = {}
    containers_dir = Path('containers')
    
    for def_file in containers_dir.glob('*/*.def'):
        container_name = def_file.parent.name
        containers[container_name] = []
        
        content = def_file.read_text()
        
        # Find micromamba install section
        in_install = False
        for line in content.split('\n'):
            if 'micromamba install' in line:
                in_install = True
                continue
            
            if in_install:
                # Stop at empty line or next section
                if not line.strip() or line.strip().startswith('#') or 'chmod' in line or 'mkdir' in line:
                    in_install = False
                    continue
                
                # Extract package name (remove version pins and backslashes)
                pkg = line.strip().replace('\\', '').strip()
                if pkg and '=' in pkg:
                    pkg = pkg.split('=')[0].strip()
                elif pkg:
                    pkg = pkg.strip()
                
                if pkg and len(pkg) < 50 and pkg not in ['-y', '-n', 'base', '-c', 'conda-forge', 'bioconda']:
                    containers[container_name].append(pkg)
    
    return containers

def analyze_package_usage(containers):
    """Count package usage across containers."""
    package_count = Counter()
    package_containers = {}
    
    for container, packages in containers.items():
        if container == 'base':
            continue
        for pkg in packages:
            package_count[pkg] += 1
            if pkg not in package_containers:
                package_containers[pkg] = []
            package_containers[pkg].append(container)
    
    return package_count, package_containers

def main():
    containers = parse_container_packages()
    package_count, package_containers = analyze_package_usage(containers)
    
    print("=" * 90)
    print("PACKAGE USAGE ANALYSIS - BioPipelines Containers")
    print("=" * 90)
    
    # Packages used by 5+ pipelines (MUST move to base)
    print("\n🔴 CRITICAL: Used by 5+ pipelines (MUST move to base):")
    print("-" * 90)
    critical = [(pkg, count) for pkg, count in package_count.items() if count >= 5]
    if critical:
        for pkg, count in sorted(critical, key=lambda x: (-x[1], x[0])):
            print(f"  {count}x  {pkg:30s} | {', '.join(sorted(package_containers[pkg]))}")
    else:
        print("  None")
    
    # Packages used by 3-4 pipelines (SHOULD move to base)
    print("\n🟡 HIGH PRIORITY: Used by 3-4 pipelines (SHOULD move to base):")
    print("-" * 90)
    high = [(pkg, count) for pkg, count in package_count.items() if 3 <= count < 5]
    if high:
        for pkg, count in sorted(high, key=lambda x: (-x[1], x[0])):
            print(f"  {count}x  {pkg:30s} | {', '.join(sorted(package_containers[pkg]))}")
    else:
        print("  None")
    
    # Packages used by 2 pipelines (CONSIDER for base)
    print("\n🟢 MEDIUM PRIORITY: Used by 2 pipelines (CONSIDER for base):")
    print("-" * 90)
    medium = [(pkg, count) for pkg, count in package_count.items() if count == 2]
    for pkg, count in sorted(medium, key=lambda x: x[0])[:15]:  # Show first 15
        print(f"  {count}x  {pkg:30s} | {', '.join(sorted(package_containers[pkg]))}")
    if len(medium) > 15:
        print(f"  ... and {len(medium) - 15} more")
    
    # Current base container
    print("\n📦 CURRENT BASE CONTAINER:")
    print("-" * 90)
    if 'base' in containers:
        for pkg in sorted(containers['base']):
            in_pipelines = package_count.get(pkg, 0)
            status = "✓" if in_pipelines >= 3 else "•"
            print(f"  {status} {pkg:30s} (used by {in_pipelines} pipelines)")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY:")
    print(f"  Total containers: {len(containers)}")
    print(f"  Pipeline containers: {len(containers) - 1}")
    print(f"  Unique packages: {len(package_count)}")
    print(f"  Packages used 5+ times: {len([c for c in package_count.values() if c >= 5])}")
    print(f"  Packages used 3-4 times: {len([c for c in package_count.values() if 3 <= c < 5])}")
    print(f"  Packages used 2 times: {len([c for c in package_count.values() if c == 2])}")
    print("=" * 90)

if __name__ == '__main__':
    main()
