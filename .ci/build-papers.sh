#!/bin/bash -x

if [[ $BUILD_PAPERS == false ]]; then
  return
fi

# Build the papers
declare -a folders=("chemical-tagging-gmm" "gmm-search")
for folder in "${folders[@]}"
do
   cd $TRAVIS_BUILD_DIR
   cd "articles/$folder"
   tectonic ms.tex --print
done


# Push to GitHub
if [ -n "$GITHUB_API_KEY" ]; then
  cd $TRAVIS_BUILD_DIR
  git checkout --orphan $TRAVIS_BRANCH-pdf
  git rm -rf .
  for folder in "${folders[@]}"
  do
      git add -f "articles/$folder/ms.pdf"
  done
  git -c user.name='travis' -c user.email='travis' commit -m "building papers"
  git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
fi
